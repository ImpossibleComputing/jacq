// Split-K (K=4) variant of the plain paged GQA decode ABI. Identical tensors/geometry to
// decode_attention_paged.jac (same Qwen2.5-7B shape, NH=28/NKV=4/HD=128/LQ=3968) -- the
// only difference is the added SPLIT_K param, which crates/jac-codegen-cuda/src/variants.rs's
// match_plain_paged_attention reads to select the split-K kernel
// (plain_paged_attention_splitk_device_kernel) instead of the SPLIT_K=1 single-warp-per-head one.
//
// Motivation: the SPLIT_K=1 kernel launches only NH=28 warps total for the whole
// kernel -- a tiny fraction of a modern GPU's warp-level capacity, and every one of those 28 SMs
// runs a single warp with nothing else resident to hide a memory load or the per-token
// warp-shuffle-reduction's dependency chain behind. That under-occupancy is an architectural
// concern, not a measured number in this repository. SPLIT_K=4 raises the warp count to
// NH*4=112, which should better hide those latencies; the actual effect is unmeasured here
// (see docs/LIMITATIONS.md and docs/GPU_VALIDATION.md). See decode_attention_paged_splitk8.jac
// for K=8.
kernel decode_attention_paged_splitk4 {

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
    SPLIT_K     = 4
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the plain paged-KV tensor ABI + SPLIT_K param above and
    // lowers this phase to the split-K fused decode kernel (emit_gathered_page_step, reused
    // per-chunk, plus a shared-memory flash-attention combine).
  }
}
