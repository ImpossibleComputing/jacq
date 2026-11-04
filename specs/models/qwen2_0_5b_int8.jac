model qwen2_0_5b_int8 {

  config {
    hidden     = 896
    layers     = 24
    heads      = 14
    kv_heads   = 2
    head_dim   = 64
    ffn_dim    = 4864
    vocab      = 151936
    max_seq    = 2048
    rope_base  = 1000000.0
    norm_eps   = 1e-6
    attn_bias  = true
  }

  weights {
    format = huggingface
    source = "Qwen/Qwen2-0.5B-Instruct"
    dtype  = int8
  }

  layer decoder {
    normed = rmsnorm(input, norm_weight, norm_eps)
    q, k, v = linear(normed, w_q, w_k, w_v)
    q, k = rope(q, k, pos, rope_base)
    attn_out = attention(q, k, v, kv_cache)
    hidden = residual(input, linear(attn_out, w_o))
    normed2 = rmsnorm(hidden, post_norm_weight, norm_eps)
    gate, up = linear(normed2, w_gate, w_up)
    ffn_mid = silu(gate) * up
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
