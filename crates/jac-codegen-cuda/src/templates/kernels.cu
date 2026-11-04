// ---- kernels ----

// Embedding lookup: out[i] = table[token * H + i].
__global__ void embed_kernel(__half* out, const __half* table, int token, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) out[i] = table[(size_t)token * hidden + i];
}

// RMSNorm with warp-shuffle reduction (fp32 math), launched as <<<1, 256>>>.
__global__ void rmsnorm_kernel(__half* out, const __half* x, const __half* weight,
                               int n, float eps) {
    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(x[i]);
        local += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(0xffffffff, local, off);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        float total = 0.0f;
        for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
        warp_sums[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(warp_sums[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(x[i]) * scale * __half2float(weight[i]));
}

// Fused RoPE for Q and K using the split-half (rotate_half) convention, matching HuggingFace
// LLaMA `apply_rotary_pos_emb`: pair (i, i + HD/2) rotates by angle pos / ROPE_BASE^(2i/HD).
// This pairs with HF-exported q/k weights directly (do NOT use the interleaved/adjacent-pair
// convention here, or the positional encoding is scrambled). Blocks [0, NH) rotate Q heads;
// blocks [NH, NH+NKV) rotate K heads.
__global__ void rope_qk_kernel(__half* q, __half* k, int pos) {
    int h = blockIdx.x;
    __half* x = (h < Config::NH) ? q + h * Config::HD
                                 : k + (h - Config::NH) * Config::HD;
    const int half = Config::HD / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(Config::ROPE_BASE, (2.0f * i) / Config::HD);
        float ang = pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = __half2float(x[i]);
        float x1 = __half2float(x[i + half]);
        x[i]        = __float2half(x0 * c - x1 * s);
        x[i + half] = __float2half(x1 * c + x0 * s);
    }
}

// Append the freshly-projected K and V (NKV heads) into this layer's cache at position `pos`.
// Cache layout per layer: [NKV, MAX_SEQ, HD].
__global__ void kv_cache_store_kernel(__half* k_cache, __half* v_cache,
                                      const __half* k_new, const __half* v_new, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Config::NKV * Config::HD;
    if (idx < total) {
        int h = idx / Config::HD;
        int d = idx % Config::HD;
        size_t off = (size_t)h * Config::MAX_SEQ * Config::HD + (size_t)pos * Config::HD + d;
        k_cache[off] = k_new[idx];
        v_cache[off] = v_new[idx];
    }
}

// scores[h*MAX_SEQ + s] = (Q[h] . K_cache[kv_h, s]) / sqrt(HD), for s in [0, pos]. One block
// per query head; GQA maps kv_h = h / HEADS_PER_GROUP.
__global__ void attention_score_kernel(const __half* q, const __half* k_cache,
                                       float* scores, int pos) {
    int h = blockIdx.x;
    int kv_h = h / Config::HEADS_PER_GROUP;
    const __half* qh = q + h * Config::HD;
    const __half* kbase = k_cache + (size_t)kv_h * Config::MAX_SEQ * Config::HD;
    float scale = rsqrtf((float)Config::HD);
    for (int s = threadIdx.x; s <= pos; s += blockDim.x) {
        const __half* ks = kbase + (size_t)s * Config::HD;
        float dot = 0.0f;
        for (int d = 0; d < Config::HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        scores[(size_t)h * Config::MAX_SEQ + s] = dot * scale;
    }
}

// In-place numerically-stable softmax over scores[h, 0..pos]. One block per head.
__global__ void softmax_kernel(float* scores, int pos) {
    int h = blockIdx.x;
    float* row = scores + (size_t)h * Config::MAX_SEQ;
    int n = pos + 1;
    int tid = threadIdx.x;
    __shared__ float red[32];

    float m = -1e30f;
    for (int s = tid; s < n; s += blockDim.x) m = fmaxf(m, row[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float mm = red[0];
        for (int i = 1; i < nw; ++i) mm = fmaxf(mm, red[i]);
        red[0] = mm;
    }
    __syncthreads();
    float mx = red[0];

    float sum = 0.0f;
    for (int s = tid; s < n; s += blockDim.x) {
        float e = expf(row[s] - mx);
        row[s] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float ss = 0.0f;
        for (int i = 0; i < nw; ++i) ss += red[i];
        red[0] = ss;
    }
    __syncthreads();
    float inv = 1.0f / red[0];
    for (int s = tid; s < n; s += blockDim.x) row[s] *= inv;
}

// out[h, d] = sum_s softmax[h, s] * V_cache[kv_h, s, d]. One block per query head.
__global__ void attention_v_kernel(__half* out, const float* scores,
                                   const __half* v_cache, int pos) {
    int h = blockIdx.x;
    int kv_h = h / Config::HEADS_PER_GROUP;
    const float* srow = scores + (size_t)h * Config::MAX_SEQ;
    const __half* vbase = v_cache + (size_t)kv_h * Config::MAX_SEQ * Config::HD;
    for (int d = threadIdx.x; d < Config::HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s <= pos; ++s)
            acc += srow[s] * __half2float(vbase[(size_t)s * Config::HD + d]);
        out[h * Config::HD + d] = __float2half(acc);
    }
}

// Fused SiLU(gate) * up.
__global__ void silu_mul_kernel(__half* out, const __half* gate, const __half* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float s = g / (1.0f + expf(-g));
        out[i] = __float2half(s * __half2float(up[i]));
    }
}

// Elementwise add (residual): out = a + b.
__global__ void vadd_kernel(__half* out, const __half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(a[i], b[i]);
}

// In-place bias add: y[i] += bias[i]. Used after the fused QKV projection for models whose
// q/k/v projections carry a bias (e.g. Qwen2); a no-op code path for LLaMA-family models.
__global__ void bias_add_kernel(__half* y, const __half* bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __hadd(y[i], bias[i]);
}

// LoRA stage A: t[r] = sum_i A[r,i] * x[i], with A row-major [rank, in_dim]. One block per rank
// row, warp-shuffle + cross-warp reduction (same shape as gemv). `t` is the rank-wide projection.
__global__ void lora_a_kernel(__half* t, const __half* A, const __half* x, int in_dim) {
    int r = blockIdx.x;
    const __half* arow = A + (size_t)r * in_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        acc += __half2float(arow[i]) * __half2float(x[i]);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        float s = 0.0f;
        for (int w = 0; w < nwarps; ++w) s += red[w];
        t[r] = __float2half(s);
    }
}

// LoRA stage B (in-place add): y[o] += scale * sum_r B[o,r] * t[r], with B row-major [out_dim,
// rank]. One thread per output row; rank is small, so the inner loop is cheap.
__global__ void lora_b_kernel(__half* y, const __half* B, const __half* t, float scale,
                              int out_dim, int rank) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    const __half* brow = B + (size_t)o * rank;
    float acc = 0.0f;
    for (int r = 0; r < rank; ++r)
        acc += __half2float(brow[r]) * __half2float(t[r]);
    y[o] = __float2half(__half2float(y[o]) + scale * acc);
}

// GPU-side argmax over logits (<<<1, 256>>>), writing one token id.
__global__ void argmax_kernel(int* out_idx, const __half* logits, int n) {
    __shared__ float s_val[32];
    __shared__ int   s_idx[32];
    int tid = threadIdx.x;
    float best = -1e30f;
    int best_i = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > best) { best = v; best_i = i; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best) { best = ov; best_i = oi; }
    }
    if ((tid & 31) == 0) { s_val[tid >> 5] = best; s_idx[tid >> 5] = best_i; }
    __syncthreads();
    if (tid == 0) {
        int nw = (blockDim.x + 31) / 32;
        for (int i = 1; i < nw; ++i)
            if (s_val[i] > s_val[0]) { s_val[0] = s_val[i]; s_idx[0] = s_idx[i]; }
        *out_idx = s_idx[0];
    }
}

