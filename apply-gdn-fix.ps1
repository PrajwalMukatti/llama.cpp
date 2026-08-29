param([string]$SourceDir = "source", [string]$RepoDir = "gdn-fix-repo")

$file = "$SourceDir\ggml\src\ggml-vulkan\ggml-vulkan.cpp"
$shader = "$SourceDir\ggml\src\ggml-vulkan\vulkan-shaders\gated_delta_net.comp"
Write-Host "=== Applying Vulkan GDN cache fusion fix ==="
Write-Host "Source: $SourceDir"

# ── STEP 1: patch ggml-vulkan.cpp ────────────────────────────────────────────
$c = Get-Content $file -Raw

# 1a. Add vk_gdn_fused_cache struct after gdn push_constants (before ssm_scan)
$structAdd = "`r`n`r`nstruct vk_gdn_fused_cache {`r`n    float *  data;`r`n    int64_t  slot_stride;`r`n    uint32_t s_off_cache;`r`n};"
$insertAfter = "    uint32_t K;`r`n};"
$ssmTarget   = "struct vk_op_ssm_scan_push_constants {"
if ($c.Contains($insertAfter)) {
    $idx = $c.IndexOf($insertAfter) + $insertAfter.Length
    # Make sure we insert before ssm_scan, not somewhere else
    $ssmIdx = $c.IndexOf($ssmTarget)
    if ($idx -lt $ssmIdx) {
        $c = $c.Insert($idx, $structAdd)
        Write-Host "Step 1a OK: added vk_gdn_fused_cache struct"
    } else {
        $c = $c.Insert($ssmIdx, "struct vk_gdn_fused_cache {`r`n    float *  data;`r`n    int64_t  slot_stride;`r`n    uint32_t s_off_cache;`r`n};`r`n`r`n")
        Write-Host "Step 1a OK: added vk_gdn_fused_cache before ssm_scan (fallback)"
    }
} elseif ($c.Contains($ssmTarget)) {
    $ssmIdx = $c.IndexOf($ssmTarget)
    $c = $c.Insert($ssmIdx, "struct vk_gdn_fused_cache {`r`n    float *  data;`r`n    int64_t  slot_stride;`r`n    uint32_t s_off_cache;`r`n};`r`n`r`n")
    Write-Host "Step 1a OK: added vk_gdn_fused_cache before ssm_scan"
} else { Write-Error "Step 1a FAILED"; exit 1 }

# 1b. Add state_out_off + state_slot_stride to push_constants struct
$pcTarget  = "    uint32_t K;`r`n};"
$pcReplace = "    uint32_t K;`r`n    uint32_t state_out_off;   // sentinel: 0=non-fused, N+1=fused with cache offset N`r`n};"
# Only replace the FIRST occurrence (the gdn push_constants, not any other struct)
$pcIdx = $c.IndexOf($pcTarget)
if ($pcIdx -ge 0) {
    $c = $c.Remove($pcIdx, $pcTarget.Length).Insert($pcIdx, $pcReplace)
    Write-Host "Step 1b OK: added state_out_off + state_slot_stride to push_constants"
} else { Write-Error "Step 1b FAILED"; exit 1 }

# 1c. Change num_bindings from 7 to 8 in ggml_vk_create_pipeline for gdn pipelines
# The create_pipeline call for gdn has "main", 7, sizeof(vk_op_gated_delta_net_push_constants)
$old7 = '"main", 7, sizeof(vk_op_gated_delta_net_push_constants)'
$new8 = '"main", 8, sizeof(vk_op_gated_delta_net_push_constants)'
if ($c.Contains($old7)) {
    $c = $c.Replace($old7, $new8)
    Write-Host "Step 1c OK: changed num_bindings 7->8"
} else { Write-Error "Step 1c FAILED: could not find pipeline creation"; exit 1 }

# 1d. Add ggml_vk_try_gdn_cache_fusion function before ggml_vk_gated_delta_net
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
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32)) return 0;
    // NOTE: dst->nb[1] is the KV cache row stride (mem_size * row_size), NOT D*sizeof(float).
    // We check src->nb[1] instead — src is a contiguous view of gdn_out with stride D.
    if (src->nb[1] != (size_t)ggml_row_size(GGML_TYPE_F32, D)) return 0;
    fc.data = (float *)dst->data;
    // slot_stride: stride between rollback slots in the KV cache (in float elements)
    // dst->nb[2] = mem_size * row_size = stride between snapshot slots
    fc.slot_stride = K > 1 ? (int64_t)(dst->nb[2] / sizeof(float)) : 0;
    // s_off_cache: byte offset of dst->data from the buffer base, in float elements.
    // Sentinel: pass (offset + 1) so the shader can use > 0 as the "fused" test
    // even when the real offset is 0 (which is valid for slot 0 at buffer start).
    const uint32_t byte_off = (uint32_t)((char *)dst->data - (char *)ggml_backend_buffer_get_base(dst->buffer));
    fc.s_off_cache = byte_off / sizeof(float) + 1u; // +1 sentinel
    return skip;
}

"@
$funcIdx = $c.IndexOf("static void ggml_vk_gated_delta_net(")
if ($funcIdx -ge 0) {
    $c = $c.Insert($funcIdx, $fusionFn)
    Write-Host "Step 1d OK: added ggml_vk_try_gdn_cache_fusion"
} else { Write-Error "Step 1d FAILED"; exit 1 }

# 1e. Update ggml_vk_gated_delta_net signature + body to accept + pass fused_cache
$oldSig = "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {"
$newSig = "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_gdn_fused_cache * fused_cache = nullptr) {"
if ($c.Contains($oldSig)) {
    $c = $c.Replace($oldSig, $newSig)
    Write-Host "Step 1e OK: updated function signature"
} else { Write-Error "Step 1e FAILED: signature not found"; exit 1 }

# 1f. In ggml_vk_gated_delta_net body: update push_constants to include state_out_off + state_slot_stride
#     and pass cache buffer as binding 7 when fused_cache != nullptr
$oldPC = @"
    const vk_op_gated_delta_net_push_constants pc = {
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
        pc, { H, n_seqs, S_v });
"@
$newPC = @"
    // Fused cache: when fused_cache != nullptr, state_out_off is the sentinel-shifted
    // element offset into the cache buffer. Shader uses state_out_off > 0 as fused test.
    uint32_t state_out_off = 0;
    vk_subbuffer cache_buf = dst_buf; // dummy binding 7 for non-fused path

    if (fused_cache != nullptr) {
        state_out_off = fused_cache->s_off_cache; // already +1 sentinel from detection
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
        pc, { H, n_seqs, S_v });
"@
if ($c.Contains($oldPC)) {
    $c = $c.Replace($oldPC, $newPC)
    Write-Host "Step 1f OK: updated push_constants + dispatch with cache binding"
} else {
    Write-Host "Step 1f WARNING: exact PC block not found, trying with LF..."
    $oldPCLF = $oldPC.Replace("`r`n", "`n")
    if ($c.Contains($oldPCLF)) {
        $c = $c.Replace($oldPCLF, $newPC)
        Write-Host "Step 1f OK (LF variant)"
    } else { Write-Error "Step 1f FAILED"; exit 1 }
}

# 1g. Add fusion dispatch in the GATED_DELTA_NET case
$callSite = $c.IndexOf("ggml_vk_gated_delta_net(ctx, compute_ctx, node);")
if ($callSite -ge 0) {
    $caseIdx = $c.LastIndexOf("case GGML_OP_GATED_DELTA_NET:", $callSite)
    $callEnd = $callSite + "ggml_vk_gated_delta_net(ctx, compute_ctx, node);".Length
    $newBlock = "case GGML_OP_GATED_DELTA_NET:" + [System.Environment]::NewLine +
                "        {" + [System.Environment]::NewLine +
                "            vk_gdn_fused_cache fc;" + [System.Environment]::NewLine +
                "            const int skip = ggml_vk_try_gdn_cache_fusion(cgraph, node_idx, fc);" + [System.Environment]::NewLine +
                "            if (skip > 0) ctx->num_additional_fused_ops = skip;" + [System.Environment]::NewLine +
                "            ggml_vk_gated_delta_net(ctx, compute_ctx, node, skip > 0 ? &fc : nullptr);" + [System.Environment]::NewLine +
                "        }"
    $c = $c.Remove($caseIdx, $callEnd - $caseIdx).Insert($caseIdx, $newBlock)
    Write-Host "Step 1g OK: added fusion dispatch"
} else { Write-Error "Step 1g FAILED: dispatch site not found"; exit 1 }

Set-Content $file $c -NoNewline
Write-Host ""
Write-Host "=== C++ patch verification ==="
Write-Host "  vk_gdn_fused_cache struct:       " $c.Contains("vk_gdn_fused_cache")
Write-Host "  state_out_off in push_constants: " $c.Contains("state_out_off")
Write-Host "  num_bindings = 8:                " $c.Contains('"main", 8, sizeof(vk_op_gated_delta_net_push_constants)')
Write-Host "  try_gdn_cache_fusion:            " $c.Contains("ggml_vk_try_gdn_cache_fusion")
Write-Host "  fusion dispatch (skip > 0):      " $c.Contains("skip > 0")

# ── STEP 2: replace the shader with the fused version ────────────────────────
Write-Host ""
Write-Host "=== Replacing GDN shader with fused version ==="
$fusedShader = "$RepoDir\gated_delta_net_fused.comp"
if (Test-Path $fusedShader) {
    Copy-Item $fusedShader $shader -Force
    Write-Host "Step 2 OK: shader replaced"
    Write-Host "  state_out_off push constant: " (Get-Content $shader -Raw).Contains("state_out_off")
    Write-Host "  CacheBuf binding 7:          " (Get-Content $shader -Raw).Contains("binding = 7")
    Write-Host "  write_state function:        " (Get-Content $shader -Raw).Contains("write_state")
} else { Write-Error "Step 2 FAILED: fused shader not found at $fusedShader"; exit 1 }

Write-Host ""
Write-Host "=== All patches applied successfully ==="
