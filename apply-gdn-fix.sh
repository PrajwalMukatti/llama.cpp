#!/usr/bin/env bash
# Bash port of apply-gdn-fix.ps1 — applies the Vulkan GDN cache-fusion patch to the REAL
# ggml-vulkan.cpp + gated_delta_net.comp at whatever llama.cpp checkout is passed as $1.
# Uses python3 for the structural edits (robust across whitespace/line-endings).
set -euo pipefail
SRC="${1:-source}"
CPP="$SRC/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
COMP="$SRC/ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp"

echo "=== Applying Vulkan GDN cache fusion to $SRC ==="

python3 - "$CPP" "$COMP" <<'PY'
import sys, re
cpp_path, comp_path = sys.argv[1], sys.argv[2]

# ---------- 1. ggml-vulkan.cpp ----------
c = open(cpp_path, encoding='utf-8').read()

# 1a. struct after gdn push_constants (before ssm_scan)
ssm = "struct vk_op_ssm_scan_push_constants {"
struct = ("struct vk_gdn_fused_cache {\n"
          "    float *  data;\n"
          "    int64_t  slot_stride;\n"
          "    uint32_t s_off_cache;\n"
          "};\n\n")
assert ssm in c, "ssm_scan struct anchor not found"
c = c.replace(ssm, struct + ssm, 1)

# 1b. add state_out_off to the gdn push_constants struct (first '    uint32_t K;\n};')
pc_anchor = "    uint32_t K;\n};"
assert pc_anchor in c, "gdn push_constants anchor not found"
c = c.replace(pc_anchor,
              "    uint32_t K;\n    uint32_t state_out_off;   // sentinel: 0=non-fused, N+1=fused with cache offset N\n};",
              1)

# 1c. num_bindings 7 -> 8
c = c.replace('"main", 7, sizeof(vk_op_gated_delta_net_push_constants)',
              '"main", 8, sizeof(vk_op_gated_delta_net_push_constants)', 1)

# 1d. fusion function (with GGML_VK_DISABLE_FUSION gate) before ggml_vk_gated_delta_net
fusion = r'''static int ggml_vk_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx, vk_gdn_fused_cache & fc) {
    static const bool disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_VK_DISABLE_FUSION")) != 0;
    if (disable_fusion) return 0;
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) return 0;
    const ggml_tensor * src_v = gdn->src[2];
    const int64_t S_v = src_v->ne[0], H = src_v->ne[1], n_tokens = src_v->ne[2], n_seqs = src_v->ne[3];
    const int64_t D = S_v * S_v * H, K = ggml_get_op_params_i32(gdn, 0);
    const int64_t n_written = std::min<int64_t>(n_tokens, K);
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);
    const ggml_tensor * cpy = nullptr; int skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && !cpy; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n) || ggml_op_is_empty(n->op) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) return 0;
        cpy = n; skip = j - node_idx;
    }
    if (!cpy) return 0;
    const ggml_tensor * src = cpy->src[0], * dst = cpy->src[1];
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off || !ggml_is_contiguous(src)) return 0;
    const std::array<int64_t, GGML_MAX_DIMS> ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || !dst->data || !dst->buffer ||
        !std::equal(ne.begin(), ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32)) return 0;
    if (src->nb[1] != (size_t)ggml_row_size(GGML_TYPE_F32, D)) return 0;
    fc.data = (float *)dst->data;
    fc.slot_stride = K > 1 ? (int64_t)(dst->nb[2] / sizeof(float)) : 0;
    const uint32_t byte_off = (uint32_t)((char *)dst->data - (char *)ggml_backend_buffer_get_base(dst->buffer));
    fc.s_off_cache = byte_off / sizeof(float) + 1u;
    return skip;
}

'''
sig = "static void ggml_vk_gated_delta_net("
i = c.index(sig)
c = c[:i] + fusion + c[i:]

# 1e. signature: add optional fused_cache param
c = c.replace(
    "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {",
    "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_gdn_fused_cache * fused_cache = nullptr) {",
    1)

# 1f. push-constants build + dispatch: add state_out_off + cache binding
old_pc = """    const vk_op_gated_delta_net_push_constants pc = {
        H, n_tokens, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });"""
new_pc = """    uint32_t state_out_off = 0;
    vk_subbuffer cache_buf = dst_buf; // dummy binding 7 on the non-fused path
    if (fused_cache != nullptr) {
        state_out_off = fused_cache->s_off_cache;
    }

    const vk_op_gated_delta_net_push_constants pc = {
        H, n_tokens, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K,
        state_out_off
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf, cache_buf},
        pc, { H, n_seqs, S_v });"""
assert old_pc in c, "push-constants/dispatch anchor not found (shader/cpp drifted from expected)"
c = c.replace(old_pc, new_pc, 1)

# 1g. dispatch case: try fusion, skip the CPY
old_case = """    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node);"""
new_case = """    case GGML_OP_GATED_DELTA_NET:
        {
            vk_gdn_fused_cache fc;
            const int skip = ggml_vk_try_gdn_cache_fusion(cgraph, node_idx, fc);
            if (skip > 0) ctx->num_additional_fused_ops = skip;
            ggml_vk_gated_delta_net(ctx, compute_ctx, node, skip > 0 ? &fc : nullptr);
        }"""
assert old_case in c, "dispatch case anchor not found"
c = c.replace(old_case, new_case, 1)

open(cpp_path, 'w', encoding='utf-8').write(c)
print("patched ggml-vulkan.cpp")

# ---------- 2. gated_delta_net.comp ----------
s = open(comp_path, encoding='utf-8').read()

# 2a. push constant: add state_out_off after K
s = s.replace("    uint K;\n};",
              "    uint K;\n    uint state_out_off; // sentinel: 0=non-fused, N+1=fused (cache elem offset N)\n};", 1)

# 2b. binding 7 (CacheBuf) after DstBuf (binding 6)
s = s.replace("layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };",
              "layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };\n"
              "layout(binding = 7)           buffer CacheBuf { FLOAT_TYPE data_cache[]; };", 1)

# 2c. K>1 slot write: route to cache when fused
old_k1 = """                const uint slot_base = s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                    data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                }"""
new_k1 = """                if (state_out_off > 0u) {
                    const uint slot_base = (state_out_off - 1u) + uint(target_slot) * state_size_per_snap + state_out_base;
                    [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                        data_cache[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                    }
                } else {
                    const uint slot_base = s_off + uint(target_slot) * state_size_per_snap + state_out_base;
                    [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {
                        data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
                    }
                }"""
assert old_k1 in s, "shader K>1 write anchor not found"
s = s.replace(old_k1, new_k1, 1)

# 2d. K==1 final write: route to cache when fused
old_k0 = """            data_dst[s_off + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];"""
new_k0 = """            if (state_out_off > 0u) {
                data_cache[(state_out_off - 1u) + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
            } else {
                data_dst[s_off + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
            }"""
assert old_k0 in s, "shader K==1 write anchor not found"
s = s.replace(old_k0, new_k0, 1)

open(comp_path, 'w', encoding='utf-8').write(s)
print("patched gated_delta_net.comp")
PY

echo "=== verification ==="
grep -q "vk_gdn_fused_cache" "$CPP" && echo "  cpp: struct OK"
grep -q "GGML_VK_DISABLE_FUSION" "$CPP" && echo "  cpp: disable-gate OK"
grep -q "binding = 7" "$COMP" && echo "  comp: cache binding OK"
grep -q "state_out_off" "$COMP" && echo "  comp: sentinel OK"
echo "=== done ==="
