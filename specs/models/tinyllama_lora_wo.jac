// TinyLlama with a LoRA adapter on the output projection (w_o) only — the first LoRA codegen
// target. `jacq --emit-model specs/tinyllama_lora_wo.jac` emits a runtime that applies
// proj += (alpha/rank) * B_o (A_o · attn) after each layer's o-projection.
model tinyllama_lora_wo {

  config {
    hidden     = 2048
    layers     = 22
    heads      = 32
    kv_heads   = 4
    head_dim   = 64
    ffn_dim    = 5632
    vocab      = 32000
    max_seq    = 2048
    rope_base  = 10000.0
    norm_eps   = 1e-5
  }

  weights {
    format = huggingface
    source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype  = fp16
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

  lora {
    rank    = 16
    alpha   = 32
    targets = w_o
    source  = "validation/random-adapter"
  }
}
