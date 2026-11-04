// Correctness check for jac-native WINDOWED decode-attention codegen (Fork 1).
//
// Both subjects use sink=4/window=512. At L=512 the window covers the whole context (no-op mask,
// = exact); at L=2048 the mask is ACTIVE (drops the middle 1532 keys) — so this exercises the
// generated attn_kept masking against the hand-written masked.cu reference at a real windowing point.
experiment decode_attention_windowed_check {
  kind = kernel
  fixture   { op = "decode_attention_windowed.jac" }
  subjects  {
    masked  = cuda("windowed_attn/masked.cu")   // hand-written windowed reference (sink=4/window=512)
    jac_win = jac(naive)                          // jac-generated windowed attention
  }
  reference = masked
  sweep     { L = [512, 2048] }
  metrics   = [correctness, latency, memory]
}
