// Fixture / ABI carrier for the windowed decode-attention perf study (track B, step 2).
// Only the tensor signature + params are consumed (emit_subject_harness): cuda() subjects
// supply the implementation, so this op is never codegen'd by jacq. The phase body is a
// parseable placeholder.
//
// Decode attention, batch=1: one query per head (NH) attends to an L-long KV cache
// (NKV kv-heads, GQA HEADS_PER_GROUP = NH/NKV). HD = head dim. Policy (S,W) is fixed in
// the subjects (sink=4, window=512). The sweep varies L.
kernel windowed_attn {
  tensors {
    Q[NH, HD]      : fp16  @global -> stream
    K[NKV, L, HD]  : fp16  @global -> stream
    V[NKV, L, HD]  : fp16  @global -> stream
    O[NH, HD]      : fp32  @registers -> global
  }
  params {
    NH  = 32
    HD  = 64
    NKV = 4
    L   = 2048
  }
  phase compute over Q.dim(NH) by NH {
    load q_tile = Q[tile, *]
  }
}
