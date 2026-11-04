// GPU correctness + latency check for the jac-native plain paged-GQA decode-attention codegen
// (the 2nd/non-KIVI consumer of `emit_gathered_page_step`). Before this experiment,
// the kernel had only ever been proven via a numpy algorithmic-parity check -- the actual
// compiled .cu had never run on a GPU. This is the first real execution.
//
// `reference` is a hand-written TWO-PASS (score/softmax/value) CUDA kernel computing the exact
// same function via a structurally different algorithm (shared/specs/paged_attn/reference.cu) --
// a genuine independent cross-check against the DSL's fused, single-warp, online-softmax
// `paged_flash` kernel, not two copies of the same recurrence wearing a correctness-check
// costume. Both read K/V through the SAME BLOCK_TABLE gather indirection (a genuinely permuted
// table -- see the harness's BLOCK_TABLE seeding fix in jac-codegen-cuda/src/experiment.rs --
// so this actually exercises the indirection, not an identity/degenerate pass-through).
//
// The kernel is fully pinned (no runtime SEQ_LEN yet, whole pages only), so there is no
// meaningful axis to sweep -- NH=[28] is the spec's own single geometry, matching
// decode_attention_check.jac's precedent of a single-value sweep for a fixed-shape check.
experiment decode_attention_paged_check {
  kind = kernel
  fixture   { op = "decode_attention_paged.jac" }
  subjects  {
    reference = cuda("paged_attn/reference.cu")
    jac_paged = jac(paged_flash)
  }
  reference = reference
  sweep     { NH = [28] }
  metrics   = [correctness, latency, memory]
}
