// Source-vLLM paged KIVI4 decode attention ABI.
//
// This is the serving-facing sibling of decode_attention_kivi.jac.  Old quantized KV is stored in
// physical vLLM pages and addressed through BLOCK_TABLE[logical_page] -> physical_page.  The first
// lane keeps the recent residual window as fp16 storage plus a 160-token dynamic scratch tail.
// KIVI only ages out full 32-token groups, so requests with 129..159 live residual tokens cannot
// be represented by a static 128-token residual tensor alone. The paged codegen threads runtime
// SEQ_LEN through jac_launch and bounds old pages by floor((SEQ_LEN - RESIDUAL) / BLOCK_SIZE), then
// attends the dynamic scratch residual tail. It is not a served-result claim: the vLLM fork must
// still bind real tensors, call this artifact, and pass CUDA-vs-kivi_pack gates.
kernel decode_attention_kivi_paged {

  tensors {
    Q[NH, HD]                                             : fp16  @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                               : int32 @global -> stream
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
    // First source-vLLM serving lane: Qwen2.5-Coder-32B attention geometry;
    // LQ+RESIDUAL = 4096. The non-paged spec stays on the
    // Qwen2.5-7B geometry, but this paged artifact must match
    // integrations.vllm_kivi.bindings.default_qwen32b_binding_abi().
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
    // Semantic marker: codegen recognizes the vLLM-paged KIVI tensor ABI and lowers to a
    // block-table-addressed fused int4 decode kernel.
  }
}
