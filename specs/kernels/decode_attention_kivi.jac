// Faithful KIVI4 decode attention ABI.
//
// KIVI storage split:
// - K_CODE/V_CODE: packed int4 codes, two head-dim values per byte.
// - K_SCALE/K_ZERO: key scale/zero grouped over tokens per channel (KIVI_GROUP=32).
// - V_SCALE/V_ZERO: value scale/zero grouped over channels per token.
// - K_RESIDUAL/V_RESIDUAL: recent residual window kept as fp16.
//
// This fixture makes the winning KIVI decode envelope DSL-expressible: fused
// score/softmax/value over quantized old KV plus fp16 residual, with scores never materialized.
kernel decode_attention_kivi {

  tensors {
    Q[NH, HD]                         : fp16 @global -> stream
    K_CODE[NKV, LQ, HD_PACKED]        : int8 @global -> stream
    K_SCALE[NKV, LQ_GROUPS, HD]       : fp16 @global -> stream
    K_ZERO[NKV, LQ_GROUPS, HD]        : fp16 @global -> stream
    V_CODE[NKV, LQ, HD_PACKED]        : int8 @global -> stream
    V_SCALE[NKV, LQ, HD_GROUPS]       : fp16 @global -> stream
    V_ZERO[NKV, LQ, HD_GROUPS]        : fp16 @global -> stream
    K_RESIDUAL[NKV, RESIDUAL, HD]     : fp16 @global -> stream
    V_RESIDUAL[NKV, RESIDUAL, HD]     : fp16 @global -> stream
    O[NH, HD]                         : fp32 @registers -> global
  }

  params {
    // Qwen2.5-7B attention geometry; LQ+RESIDUAL = 4096.
    NH          = 28
    NKV         = 4
    HD          = 128
    LQ          = 3968
    RESIDUAL    = 128
    KIVI_GROUP  = 32
    HD_PACKED   = 64
    LQ_GROUPS   = 124
    HD_GROUPS   = 4
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the KIVI tensor ABI above and lowers this phase to the
    // fused int4 decode kernel. The explicit dequant op becomes first-class DSL syntax later.
  }
}
