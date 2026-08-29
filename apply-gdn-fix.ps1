param([string]$SourceDir = "source")

$file = "$SourceDir\ggml\src\ggml-vulkan\ggml-vulkan.cpp"
Write-Host "Patching: $file"
$c = Get-Content $file -Raw

# 1 - Add vk_gdn_fused_cache struct after push_constants
$structTarget = "static_assert(sizeof(vk_op_gated_delta_net_push_constants) <= 128);"
$structAdd = @"

struct vk_gdn_fused_cache {
    float *  data;
    int64_t  slot_stride;
    uint32_t s_off_cache;
};
"@
if ($c.Contains($structTarget)) {
    $c = $c.Replace($structTarget, $structTarget + $structAdd)
    Write-Host "Step 1 OK: added vk_gdn_fused_cache"
} else { Write-Error "Step 1 FAILED: target not found"; exit 1 }

# 2 - Add fusion function before ggml_vk_gated_delta_net
$fusionFn = @"
static int ggml_vk_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx, vk_gdn_fused_cache & fc) {
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
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t)ggml_row_size(GGML_TYPE_F32, D)) return 0;
    fc.data = (float *)dst->data;
    fc.slot_stride = K > 1 ? (int64_t)(dst->nb[2] / sizeof(float)) : 0;
    fc.s_off_cache = (uint32_t)((char *)dst->data - (char *)ggml_backend_buffer_get_base(dst->buffer));
    return skip;
}

"@
$funcTarget = "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {"
if ($c.Contains($funcTarget)) {
    $c = $c.Replace($funcTarget, $fusionFn + $funcTarget)
    Write-Host "Step 2 OK: added ggml_vk_try_gdn_cache_fusion"
} else { Write-Error "Step 2 FAILED: gated_delta_net function not found"; exit 1 }

# 3 - Update signature to accept optional cache
$oldSig = "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {"
$newSig = "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_gdn_fused_cache * fused_cache = nullptr) {"
$c = $c.Replace($oldSig, $newSig)
Write-Host "Step 3 OK: updated signature"

# 4 - Add fusion check in dispatch case
$oldDispatch = "    case GGML_OP_GATED_DELTA_NET:`n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);"
$newDispatch = @"
    case GGML_OP_GATED_DELTA_NET:
        {
            vk_gdn_fused_cache fc;
            const int skip = ggml_vk_try_gdn_cache_fusion(cgraph, node_idx, fc);
            if (skip > 0) ctx->num_additional_fused_ops = skip;
            ggml_vk_gated_delta_net(ctx, compute_ctx, node, skip > 0 ? &fc : nullptr);
        }
"@
# Try both LF and CRLF versions
if ($c.Contains("    case GGML_OP_GATED_DELTA_NET:`r`n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);")) {
    $c = $c.Replace("    case GGML_OP_GATED_DELTA_NET:`r`n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);", $newDispatch)
    Write-Host "Step 4 OK: added fusion dispatch (CRLF)"
} elseif ($c.Contains("    case GGML_OP_GATED_DELTA_NET:`n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);")) {
    $c = $c.Replace("    case GGML_OP_GATED_DELTA_NET:`n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);", $newDispatch)
    Write-Host "Step 4 OK: added fusion dispatch (LF)"
} else {
    Write-Host "Step 4 WARNING: dispatch pattern not found - checking context..."
    $idx = $c.IndexOf("GGML_OP_GATED_DELTA_NET")
    if ($idx -ge 0) { Write-Host "Found at index $idx :" $c.Substring($idx, [Math]::Min(200, $c.Length - $idx)) }
}

Set-Content $file $c -NoNewline
Write-Host ""
Write-Host "Verification:"
Write-Host "  vk_gdn_fused_cache:          " $c.Contains("vk_gdn_fused_cache")
Write-Host "  ggml_vk_try_gdn_cache_fusion:" $c.Contains("ggml_vk_try_gdn_cache_fusion")
Write-Host "  fused_cache optional param:  " $c.Contains("fused_cache = nullptr")
Write-Host "  fusion dispatch:             " $c.Contains("ggml_vk_try_gdn_cache_fusion(cgraph")
Write-Host "Patch complete."
