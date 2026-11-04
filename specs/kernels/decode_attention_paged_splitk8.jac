// Split-K (K=8) variant of the plain paged GQA decode ABI. See
// decode_attention_paged_splitk4.jac for the full motivation/design doc -- this is the same spec
// with SPLIT_K=8 instead of 4, the other end of the candidate warp-count range.
// Same tensors/geometry otherwise (Qwen2.5-7B shape, NH=28/NKV=4/HD=128/LQ=3968).
kernel decode_attention_paged_splitk8 {

  tensors {
    Q[NH, HD]                                  : fp16  @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                    : int32 @global -> stream
    K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD]   : fp16  @global -> stream
    V_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD]   : fp16  @global -> stream
    O[NH, HD]                                  : fp32  @registers -> global
  }

  params {
    NH          = 28
    NKV         = 4
    HD          = 128
    LQ          = 3968
    BLOCK_SIZE  = 32
    NUM_BLOCKS  = 124
    MAX_BLOCKS  = 124
    SPLIT_K     = 8
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the plain paged-KV tensor ABI + SPLIT_K param above and
    // lowers this phase to the split-K fused decode kernel (emit_gathered_page_step, reused
    // per-chunk, plus a shared-memory flash-attention combine).
  }
}
