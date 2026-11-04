// Source-vLLM BATCHED paged KIVI4 decode attention ABI (batched prefill-loop fix).
//
// Batched sibling of decode_attention_kivi_paged.jac. The single-token kernel is completely
// untouched by this spec -- this is a new, separate kernel name/ABI, not a modification.
//
// Design: the single-token kernel already computes one query's online-softmax flash-decode
// attention against the full (quantized-pages + fp16-residual-scratch) KV state for that query's
// own SEQ_LEN. Reasoned from source_patch.py's actual request loop: a prefill chunk of N
// new tokens needs N of these per-query computations, one per token, each against that token's own
// SEQ_LEN (start_seq_len + i + 1 for the i-th new token) -- today issued as N sequential kernel
// launches. This spec batches them into ONE launch with a 2D grid (heads x queries) instead of N
// launches; the per-query math is UNCHANGED (same online-softmax recurrence, same page/residual
// addressing), so each query's own output is expected to be bit-identical to what the single-token
// kernel already produces for that (head, seq_len) pair -- this is a launch-count reduction, not a
// new correctness surface, and must be GPU-bitchecked against the single-token kernel before any
// correctness claim (not yet done as of this spec landing).
//
// MAX_QUERIES bounds the batch to the same 160-token sub-chunk size the write-side
// batching plan (plan_batched_write in the scaffolded kivi_dsl.py) already uses --
// decode batching piggybacks on the exact sub-chunk boundaries the write plan
// computes (each sub-chunk's tokens are decoded together right after that
// sub-chunk's writes land), so this is not a new constant to independently justify.
kernel decode_attention_kivi_paged_batched {

  tensors {
    Q[MAX_QUERIES, NH, HD]                                : fp16  @global -> stream
    SEQ_LENS[MAX_QUERIES]                                 : int32 @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                               : int32 @global -> stream
    K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8  @global -> stream
    K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16  @global -> stream
    K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16  @global -> stream
    V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8  @global -> stream
    V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16  @global -> stream
    V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16  @global -> stream
    K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16  @global -> stream
    V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16  @global -> stream
    O[MAX_QUERIES, NH, HD]                                : fp32  @registers -> global
  }

  params {
    // Same Qwen2.5-Coder-32B geometry as the single-token paged spec (must match
    // integrations.vllm_kivi.bindings.default_qwen32b_binding_abi()).
    NH          = 40
    NKV         = 8
    HD          = 128
    LQ          = 3968
    RESIDUAL    = 128
    RESIDUAL_SCRATCH = 160
    KIVI_GROUP  = 32
    BLOCK_SIZE  = 32
    NUM_BLOCKS  = 124
    MAX_BLOCKS  = 124
    HD_PACKED   = 64
    HD_GROUPS   = 4
    // Matches the write-side batch planner's scratch-capacity sub-chunk bound
    // (RESIDUAL + BLOCK_SIZE = 160), not an independently-chosen constant.
    MAX_QUERIES = 160
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the vLLM-paged BATCHED KIVI tensor ABI (an extra
    // MAX_QUERIES leading dimension on Q/O, plus a SEQ_LENS tensor replacing the single-token
    // spec's scalar runtime SEQ_LEN) and lowers to a 2D-grid (heads x queries) fused int4 decode
    // kernel reusing the single-token kernel's own per-query online-softmax math unchanged.
  }
}
