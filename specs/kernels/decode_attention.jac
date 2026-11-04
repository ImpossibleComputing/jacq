// Decode attention (batch=1): one query per head attends to an L-long KV cache.
// jac-native codegen target — the kernel-DSL expression of the model runtime's validated
// attention (docs/PROGRESS.md: "fuse load Q/K/V -> einsum scores -> softmax -> weighted-V").
//
// GQA: NH query heads share NKV kv-heads (HEADS_PER_GROUP = NH / NKV). HD = head dim, L = context.
// This BASE variant computes EXACT attention (reads all L) — the correctness reference point.
// Windowing (sink+window), compaction, and single-pass fusion are additive variants on top.
kernel decode_attention {

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
    scores = einsum("hd, hsd -> hs", Q, K)
    probs  = softmax(scores, dim: L)
    O      = einsum("hs, hsd -> hd", probs, V)
  }
}
