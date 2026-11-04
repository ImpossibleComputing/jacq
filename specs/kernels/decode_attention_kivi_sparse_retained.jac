// Source-vLLM sparse-retained paged KIVI4 decode attention ABI.
//
// This is the page-aware-eviction conversion-path sibling of decode_attention_kivi_paged.jac.
// Old quantized KV still lives in physical vLLM pages, but the decode path no longer receives the
// dense BLOCK_TABLE covering every old logical page. Instead RETAINED_BLOCK_TABLE is compact:
// entry i is the physical page id for the i-th retained old page after eviction. The generated
// kernel loops over NUM_RETAINED_PAGES * BLOCK_SIZE old tokens, so dropped pages are not merely
// allocator-free metadata; they are skipped by attention work. SEQ_LEN still describes the
// original dense request length so the residual scratch window remains the same recent tail.
//
// This spec is source/CPU readiness only until a GPU bitcheck and goodput probe validate the
// emitted kernel against the sparse CPU reference on the served host.
kernel decode_attention_kivi_sparse_retained {

  tensors {
    Q[NH, HD]                                             : fp16  @global -> stream
    RETAINED_BLOCK_TABLE[MAX_BLOCKS]                      : int32 @global -> stream
    K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8  @global -> stream
    K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16  @global -> stream
    K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16  @global -> stream
    V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8  @global -> stream
    V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16  @global -> stream
    V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16  @global -> stream
    K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16  @global -> stream
    V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16  @global -> stream
    O[NH, HD]                                             : fp32  @registers -> global
  }

  params {
    // Same Qwen2.5-Coder-32B geometry as the dense paged KIVI serving lane.
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
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the sparse-retained vLLM-paged KIVI tensor ABI and
    // lowers to a compact-retained-page fused int4 decode kernel.
  }
}
