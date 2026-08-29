param([string]$SourceDir = "source")

$file = "$SourceDir\ggml\src\ggml-vulkan\ggml-vulkan.cpp"
Write-Host "Patching: $file"
$c = Get-Content $file -Raw

# 1 - Add vk_gdn_fused_cache struct after the push_constants closing brace
# The struct ends with "};" followed by blank line + "struct vk_op_ssm_scan_push_constants"
# Try multiple targets for compatibility across builds
$structAdd = @"

struct vk_gdn_fused_cache {
    float *  data;
    int64_t  slot_stride;
    uint32_t s_off_cache;
};

"@
# Find the gdn push_constants struct and insert after its closing };
# The struct ends with "    uint32_t K;`n};" — unique enough
$targets = @(
    "    uint32_t K;`r`n};`r`n`r`nstruct vk_op_ssm_scan_push_constants",
    "    uint32_t K;`n};`n`nstruct vk_op_ssm_scan_push_constants",
    "static_assert(sizeof(vk_op_gated_delta_net_push_constants) <= 128);"
)
$step1done = $false
foreach ($t in $targets) {
    if ($c.Contains($t)) {
        # Insert after the closing }; of gdn push_constants (before ssm_scan)
        $insertAfter = "    uint32_t K;" + [System.Environment]::NewLine + "};"
        if ($c.Contains($insertAfter)) {
            $insertIdx = $c.IndexOf($insertAfter) + $insertAfter.Length
            $c = $c.Insert($insertIdx, $structAdd)
            Write-Host "Step 1 OK: inserted vk_gdn_fused_cache after gdn push_constants"
            $step1done = $true
            break
        }
    }
}
if (-not $step1done) {
    # Fallback: insert before the ssm_scan struct
    $ssmIdx = $c.IndexOf("struct vk_op_ssm_scan_push_constants {")
    if ($ssmIdx -ge 0) {
        $c = $c.Insert($ssmIdx, $structAdd)
        Write-Host "Step 1 OK: inserted before ssm_scan (fallback)"
        $step1done = $true
    }
}
if (-not $step1done) { Write-Error "Step 1 FAILED: no insertion point found"; exit 1 }

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
$funcTargets = @(
    "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {",
    "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor *`r`n"
)
$step2done = $false
foreach ($ft in $funcTargets) {
    if ($c.Contains($ft)) {
        $c = $c.Replace($ft, $fusionFn + $ft)
        Write-Host "Step 2 OK: added ggml_vk_try_gdn_cache_fusion"
        $step2done = $true; break
    }
}
# Try prefix match
if (-not $step2done) {
    $idx = $c.IndexOf("static void ggml_vk_gated_delta_net(")
    if ($idx -ge 0) {
        $end = $c.IndexOf("{", $idx) + 1
        $orig = $c.Substring($idx, $end - $idx)
        $c = $c.Remove($idx, $end - $idx).Insert($idx, $fusionFn + $orig)
        Write-Host "Step 2 OK: added via prefix match"
        $step2done = $true
    }
}
if (-not $step2done) { Write-Error "Step 2 FAILED: function not found"; exit 1 }

# 3 - Update signature to accept optional cache (use prefix match)
$sigIdx = $c.IndexOf("static void ggml_vk_gated_delta_net(")
if ($sigIdx -ge 0) {
    $sigEnd = $c.IndexOf("{", $sigIdx) + 1
    $oldSig = $c.Substring($sigIdx, $sigEnd - $sigIdx)
    # Find the closing paren of the params and insert before the {
    $parenClose = $c.LastIndexOf(")", $sigEnd)
    $newSig = $oldSig.Substring(0, $parenClose - $sigIdx) + ", const vk_gdn_fused_cache * fused_cache = nullptr" + $oldSig.Substring($parenClose - $sigIdx)
    $c = $c.Remove($sigIdx, $sigEnd - $sigIdx).Insert($sigIdx, $newSig)
    Write-Host "Step 3 OK: updated signature"
} else { Write-Error "Step 3 FAILED"; exit 1 }

# 4 - Add fusion check in dispatch case (find by searching for the call site)
$callIdx = $c.IndexOf("ggml_vk_gated_delta_net(ctx, compute_ctx, node);")
if ($callIdx -ge 0) {
    # Find the "case GGML_OP_GATED_DELTA_NET:" line before this call
    $caseIdx = $c.LastIndexOf("case GGML_OP_GATED_DELTA_NET:", $callIdx)
    $callEnd = $callIdx + "ggml_vk_gated_delta_net(ctx, compute_ctx, node);".Length
    $origBlock = $c.Substring($caseIdx, $callEnd - $caseIdx)
    $newBlock = "case GGML_OP_GATED_DELTA_NET:" + [System.Environment]::NewLine +
                "        {" + [System.Environment]::NewLine +
                "            vk_gdn_fused_cache fc;" + [System.Environment]::NewLine +
                "            const int skip = ggml_vk_try_gdn_cache_fusion(cgraph, node_idx, fc);" + [System.Environment]::NewLine +
                "            if (skip > 0) ctx->num_additional_fused_ops = skip;" + [System.Environment]::NewLine +
                "            ggml_vk_gated_delta_net(ctx, compute_ctx, node, skip > 0 ? &fc : nullptr);" + [System.Environment]::NewLine +
                "        }"
    $c = $c.Remove($caseIdx, $callEnd - $caseIdx).Insert($caseIdx, $newBlock)
    Write-Host "Step 4 OK: added fusion dispatch"
} else { Write-Host "Step 4 WARNING: call site not found (may already be patched)" }

Set-Content $file $c -NoNewline
Write-Host ""
Write-Host "Verification:"
Write-Host "  vk_gdn_fused_cache:          " $c.Contains("vk_gdn_fused_cache")
Write-Host "  ggml_vk_try_gdn_cache_fusion:" $c.Contains("ggml_vk_try_gdn_cache_fusion")
Write-Host "  fused_cache optional param:  " $c.Contains("fused_cache = nullptr")
Write-Host "  fusion dispatch:             " $c.Contains("ggml_vk_try_gdn_cache_fusion(cgraph")
Write-Host "Patch complete."
