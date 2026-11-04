// Source-vLLM paged KIVI4 cache-writer ABI (DSL-ize the write side).
//
// Write-side sibling of decode_attention_kivi_paged.jac. Where the decode spec reads the paged
// quantized KV + fp16 residual scratch to compute attention, this spec appends one new token's K/V
// into the residual scratch and, when the scratch has grown past RESIDUAL tokens, quantizes the
// oldest 32-token group into a physical page (addressed through BLOCK_TABLE, same real
// vLLM-scheduler-assigned table the decode kernel reads) and shifts the remaining scratch left.
//
// K and V use genuinely different quantization axes, matching the KIVI paper's own asymmetry, not
// a style choice: K is quantized per-channel-pair over the aged-out 32-token window (min/max
// reduced over time, one scale/zero per channel), V is quantized per-token-per-channel-group
// (min/max reduced over a 32-wide channel group at a fixed time step, one scale/zero per
// token-group). A single shared quantization kernel would be numerically wrong for one of the two.
//
// This is the single-token write lane (one new token per call) -- the served-path writer as of
// the DSL writer landing. The batched sibling (integrations/vllm_kivi/source_patch.py's
// kivi_cache_write_batched_launch, prefill-batch fix) is a separate, not-yet-DSL-ized
// kernel; DSL-izing it is a natural follow-on once this single-token lane is verified, not
// bundled into this spec.
//
// Not a served-result claim on its own: codegen emits a CUDA library artifact from this spec, but
// it only becomes the runtime's source of truth once GPU-bitchecked byte-identical against the
// existing hand-CUDA kivi_cache_write_launch (integrations/vllm_kivi/source_patch.py) across
// representative seq_len values (with and without an age-out this call).
kernel cache_write_kivi_paged {

  tensors {
    NEW_K[NKV, HD]                                        : fp16  @global -> stream
    NEW_V[NKV, HD]                                         : fp16  @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                                : int32 @global -> stream
    K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]   : int8  @global -> stream -> global
    K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16  @global -> stream -> global
    K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                      : fp16  @global -> stream -> global
    V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]   : int8  @global -> stream -> global
    V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16  @global -> stream -> global
    V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]   : fp16  @global -> stream -> global
    K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]          : fp16  @global -> stream -> global
    V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]          : fp16  @global -> stream -> global
  }

  params {
    // Same Qwen2.5-Coder-32B geometry as the paged decode specs (must match
    // integrations.vllm_kivi.bindings.default_qwen32b_binding_abi()).
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
  }

  phase cache_write over NEW_K.dim(NKV) by NKV {
    // Semantic marker: codegen recognizes the vLLM-paged KIVI cache-writer tensor ABI (new-token
    // K/V, BLOCK_TABLE, quantized pages, fp16 residual scratch -- no Q/O attention tensors) and
    // lowers to: append the new token into residual scratch, then conditionally quantize the
    // aged-out 32-token group into its BLOCK_TABLE-addressed physical page and shift the scratch,
    // matching kivi_cache_write_launch's own runtime seq_len-driven conditional exactly.
  }
}
