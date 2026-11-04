model mistral_7b_short_context {

  // This exact-attention example stays below Mistral v0.1's 4096-token sliding window.
  config {
    hidden     = 4096
    layers     = 32
    heads      = 32
    kv_heads   = 8
    head_dim   = 128
    ffn_dim    = 14336
    vocab      = 32000
    max_seq    = 2048
    rope_base  = 10000.0
    norm_eps   = 1e-5
  }

  weights {
    format = huggingface
    source = "mistralai/Mistral-7B-v0.1"
    dtype  = fp16
  }

  layer decoder {
    // Pre-attention norm
    normed = rmsnorm(input, norm_weight, norm_eps)

    // QKV projection (fused into single cuBLAS call)
    q, k, v = linear(normed, w_q, w_k, w_v)

    // Rotary positional embedding
    q, k = rope(q, k, pos, rope_base)

    // GQA attention with KV cache
    attn_out = attention(q, k, v, kv_cache)

    // Output projection + residual
    hidden = residual(input, linear(attn_out, w_o))

    // Post-attention norm
    normed2 = rmsnorm(hidden, post_norm_weight, norm_eps)

    // FFN: gate + up projection (fused into single cuBLAS call)
    gate, up = linear(normed2, w_gate, w_up)

    // SiLU activation + elementwise multiply (fused kernel)
    ffn_mid = silu(gate) * up

    // Down projection + residual
    output = residual(hidden, linear(ffn_mid, w_down))
  }

  pipeline {
    embed(token, embed_weights)
    -> repeat(decoder, layers)
    -> rmsnorm(final_norm_weight, norm_eps)
    -> lm_head(lm_head_weights)
    -> greedy_sample
  }
}
