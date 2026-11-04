// Windowed decode attention: same shape as decode_attention.jac, but with a sink+window policy.
// The SINK/WINDOW params make match_attention emit the masked variant (drop the middle of the KV
// cache to -INF: keep the first SINK tokens + the last WINDOW tokens). This is the kernel-DSL
// expression of the hand-written masked.cu — i.e. jacq now GENERATES the windowed subject.
//
// This spec fixes sink=4 and window=512; L is the context length.
kernel decode_attention_windowed {

  tensors {
    Q[NH, HD]      : fp16  @global -> stream
    K[NKV, L, HD]  : fp16  @global -> stream
    V[NKV, L, HD]  : fp16  @global -> stream
    O[NH, HD]      : fp32  @registers -> global
  }

  params {
    NH     = 32
    HD     = 64
    NKV    = 4
    L      = 2048
    SINK   = 4
    WINDOW = 512
  }

  phase compute over Q.dim(NH) by NH {
    scores = einsum("hd, hsd -> hs", Q, K)
    probs  = softmax(scores, dim: L)
    O      = einsum("hs, hsd -> hd", probs, V)
  }
}
