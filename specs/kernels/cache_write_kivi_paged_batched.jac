// Source-vLLM BATCHED paged KIVI4 cache-writer ABI.
//
// Batched write-side sibling of cache_write_kivi_paged.jac. NEW_K/NEW_V are token-major
// [MAX_NEW, NKV, HD] slices from vLLM's per-request K/V tensors, matching the served
// JAC_KIVI_BATCHED_PREFILL=1 hand-CUDA path's kivi_cache_write_batched_launch ABI.
//
// Semantics: append NUM_NEW tokens into residual scratch starting from START_SEQ_LEN's live tail,
// then, if START_SEQ_LEN + NUM_NEW crosses exactly one age-out boundary, quantize the oldest
// 32-token residual group into BLOCK_TABLE[logical_page] and shift the remaining live residual.
// The caller must split larger prefill chunks with the batched-write planner
// (plan_batched_write in the scaffolded kivi_dsl.py) so each call ages out at
// most one page.
//
// This spec is a DSL-authored comparison artifact only until GPU-bitchecked exactly against the
// existing hand-CUDA kivi_cache_write_batched_launch over a multi-subchunk schedule.
kernel cache_write_kivi_paged_batched {

  tensors {
    NEW_K[MAX_NEW, NKV, HD]                              : fp16  @global -> stream
    NEW_V[MAX_NEW, NKV, HD]                              : fp16  @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                              : int32 @global -> stream
    K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED] : int8  @global -> stream -> global
    K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                   : fp16  @global -> stream -> global
    K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16  @global -> stream -> global
    V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED] : int8  @global -> stream -> global
    V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]: fp16  @global -> stream -> global
    V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16  @global -> stream -> global
    K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]        : fp16  @global -> stream -> global
    V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]        : fp16  @global -> stream -> global
  }

  params {
    NKV         = 8
    HD          = 128
    RESIDUAL    = 128
    RESIDUAL_SCRATCH = 160
    KIVI_GROUP  = 32
    BLOCK_SIZE  = 32
    NUM_BLOCKS  = 124
    MAX_BLOCKS  = 124
    HD_PACKED   = 64
    HD_GROUPS   = 4
    MAX_NEW     = 160
  }

  phase cache_write_batched over NEW_K.dim(MAX_NEW) by MAX_NEW {
    // Semantic marker: codegen recognizes the batched vLLM-paged KIVI cache-writer tensor ABI
    // and lowers to jac_launch_cache_write_batched, a DSL-authored sibling of the existing
    // hand-CUDA kivi_cache_write_batched_launch. Runtime START_SEQ_LEN/NUM_NEW drive append and
    // optional age-out exactly as in the served batched prefill path.
  }
}
