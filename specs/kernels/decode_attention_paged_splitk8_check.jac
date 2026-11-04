// GPU correctness + latency check for the split-K (K=8) plain paged-GQA decode kernel.
// See decode_attention_paged_splitk4_check.jac for the full doc -- identical except SPLIT_K=8.
experiment decode_attention_paged_splitk8_check {
  kind = kernel
  fixture   { op = "decode_attention_paged_splitk8.jac" }
  subjects  {
    reference         = cuda("paged_attn/reference.cu")
    jac_paged_splitk8 = jac(paged_flash)
  }
  reference = reference
  sweep     { NH = [28] }
  metrics   = [correctness, latency, memory]
}
