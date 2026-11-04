// GPU correctness + latency check for the split-K (K=4) plain paged-GQA decode kernel.
// Sibling of decode_attention_paged_check.jac (which checks the SPLIT_K=1 kernel) -- same
// hand-CUDA two-pass reference (shared/specs/paged_attn/reference.cu), same geometry, so the
// latency numbers from both checks are directly comparable: this is the A/B the
// under-occupancy finding predicts should narrow (ideally close) the ~1.5x gap measured for
// SPLIT_K=1 against the same reference.
//
// Fully pinned (no runtime SEQ_LEN, whole pages only, one SPLIT_K value per spec) -- no
// meaningful axis to sweep, matching decode_attention_paged_check.jac's own precedent.
experiment decode_attention_paged_splitk4_check {
  kind = kernel
  fixture   { op = "decode_attention_paged_splitk4.jac" }
  subjects  {
    reference         = cuda("paged_attn/reference.cu")
    jac_paged_splitk4 = jac(paged_flash)
  }
  reference = reference
  sweep     { NH = [28] }
  metrics   = [correctness, latency, memory]
}
