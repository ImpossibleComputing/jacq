// Correctness check for jac-native decode-attention codegen (Fork 1).
//
// At L=512 the hand-written masked.cu reference computes EXACT attention: its policy is
// sink=4/window=512, so kept(s) = (s<4) || (s >= L-512) = (s >= 0) is always true — nothing is
// dropped (the "window covers the whole context" control from the step-2 perf study). So the
// jac-generated EXACT kernel (`jac_gen`) must match `masked` here. They WOULD diverge at L>516
// (masked goes windowed, jac stays exact) — that divergence is correct, not a bug, so we only
// check the L where the two compute the same function.
experiment decode_attention_check {
  kind = kernel
  fixture   { op = "decode_attention.jac" }
  subjects  {
    masked  = cuda("windowed_attn/masked.cu")   // EXACT attention at L<=516 — the reference
    jac_gen = jac(naive)                          // jac-native exact decode attention (match_attention)
  }
  reference = masked
  sweep     { L = [512] }
  metrics   = [correctness, latency, memory]
}
