// Plain (non-quantized) paged decode-attention ABI -- the second real consumer of
// emit_gathered_page_step, proving the gather-indexed codegen extraction
// generalizes beyond KIVI.
//
// Same GQA-mapped score/softmax/value math as the base `decode_attention.jac` exact kernel
// (kv_head = head / (NH/NKV), any NH % NKV == 0), but K/V are gathered from physical pages via
// BLOCK_TABLE[logical_page] -> physical_page instead of read from one contiguous [NKV,L,HD]
// tensor. Unlike the KIVI paged sibling (decode_attention_kivi_paged.jac) there is no int4 unpack
// and no residual/quantization-boundary split -- every page is uniformly fp16, so this is a
// simpler consumer of the same gather primitive, not a harder one.
//
// Geometry uses the Qwen2.5-7B attention shape (NH=28, NKV=4 -> GQA group_size 7), a
// ratio FlashInfer's paged decode cannot serve natively without q-head padding.
// First slice deliberately requires LQ % BLOCK_SIZE == 0 (whole pages only, no partial last page)
// and attends the full LQ every call (no runtime SEQ_LEN yet, unlike the KIVI paged sibling) --
// that dynamic-length support is real, separate future work, not needed to prove the gather
// primitive itself generalizes to a non-KIVI caller.
// Note: BLOCK_TABLE/K_PAGES/V_PAGES use `-> stream` (not `-> gather`), matching the established
// KIVI paged sibling's convention. `-> gather` is real grammar but currently unread by codegen
// (the matcher below keys on tensor name/shape, not the access annotation) -- using `-> stream`
// avoids implying a distinction the codegen doesn't actually make.
kernel decode_attention_paged {

  tensors {
    Q[NH, HD]                                  : fp16  @global -> stream
    BLOCK_TABLE[MAX_BLOCKS]                    : int32 @global -> stream
    K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD]   : fp16  @global -> stream
    V_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD]   : fp16  @global -> stream
    O[NH, HD]                                  : fp32  @registers -> global
  }

  params {
    // Qwen2.5-7B attention geometry (LQ=3968), reused here paged instead of contiguous.
    NH          = 28
    NKV         = 4
    HD          = 128
    LQ          = 3968
    BLOCK_SIZE  = 32
    NUM_BLOCKS  = 124
    MAX_BLOCKS  = 124
  }

  phase fused_decode over Q.dim(NH) by NH {
    // Semantic marker: codegen recognizes the plain paged-KV tensor ABI above and lowers this
    // phase to a fused, gather-indexed online-softmax decode kernel (emit_gathered_page_step).
  }
}
