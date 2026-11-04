// ---- runtime ----

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// y[out_dim] = W[out_dim, in_dim] @ x[in_dim], via cuBLAS (GEMV expressed as GEMM with N=1).
// W is row-major [out_dim, in_dim]; in column-major terms that is [in_dim, out_dim] with
// leading dim in_dim, so OP_T gives the [out_dim, in_dim] @ x product. fp16 in/out, fp32 accum,
// tensor-core path.
static inline void gemv(cublasHandle_t handle, const __half* W, const __half* x, __half* y,
                        int in_dim, int out_dim) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 out_dim, 1, in_dim,
                 &alpha,
                 W, CUDA_R_16F, in_dim,
                 x, CUDA_R_16F, in_dim,
                 &beta,
                 y, CUDA_R_16F, out_dim,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

struct LayerWeights {
    __half *attn_norm, *qkv, *o, *ffn_norm, *gate_up, *down;
    __half *qkv_bias;            // q/k/v projection bias (length QKV_OUT); only when Config::ATTN_BIAS
    // LoRA adapters (A[rank, in], B[out, rank]); each present only when its Config::LORA_* flag is set.
    __half *wq_lora_a, *wq_lora_b;  // w_q: A[rank, H],     B[Q_DIM, rank]  -> qkv[0 .. Q_DIM)
    __half *wk_lora_a, *wk_lora_b;  // w_k: A[rank, H],     B[KV_DIM, rank] -> qkv[Q_DIM .. +KV_DIM)
    __half *wv_lora_a, *wv_lora_b;  // w_v: A[rank, H],     B[KV_DIM, rank] -> qkv[Q_DIM+KV_DIM ..)
    __half *wo_lora_a, *wo_lora_b;  // w_o: A[rank, Q_DIM], B[H, rank]      -> proj
};
struct Weights {
    __half *embed, *final_norm, *lm_head;
    LayerWeights layers[Config::L];
};

// Read a raw fp16 binary of `count` elements from `dir/name` and copy it to the GPU.
static __half* load_bin(const char* dir, const char* name, size_t count) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) { fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}

static void load_weights(Weights* w, const char* dir) {
    using C = Config;
    w->embed      = load_bin(dir, "embed.bin",      (size_t)C::VOCAB * C::H);
    w->final_norm = load_bin(dir, "final_norm.bin", (size_t)C::H);
    w->lm_head    = load_bin(dir, "lm_head.bin",    (size_t)C::VOCAB * C::H);
    for (int l = 0; l < C::L; ++l) {
        char nm[64];
        snprintf(nm, sizeof(nm), "layer%02d_attn_norm.bin", l); w->layers[l].attn_norm = load_bin(dir, nm, (size_t)C::H);
        snprintf(nm, sizeof(nm), "layer%02d_qkv.bin", l);       w->layers[l].qkv       = load_bin(dir, nm, (size_t)C::QKV_OUT * C::H);
        if (C::ATTN_BIAS) {
            snprintf(nm, sizeof(nm), "layer%02d_qkv_bias.bin", l); w->layers[l].qkv_bias = load_bin(dir, nm, (size_t)C::QKV_OUT);
        } else {
            w->layers[l].qkv_bias = nullptr;
        }
        LayerWeights& lw = w->layers[l];
        lw.wq_lora_a = lw.wq_lora_b = lw.wk_lora_a = lw.wk_lora_b = nullptr;
        lw.wv_lora_a = lw.wv_lora_b = lw.wo_lora_a = lw.wo_lora_b = nullptr;
        if (C::LORA_Q) {
            snprintf(nm, sizeof(nm), "layer%02d_wq_loraA.bin", l); lw.wq_lora_a = load_bin(dir, nm, (size_t)C::LORA_RANK * C::H);
            snprintf(nm, sizeof(nm), "layer%02d_wq_loraB.bin", l); lw.wq_lora_b = load_bin(dir, nm, (size_t)C::Q_DIM * C::LORA_RANK);
        }
        if (C::LORA_K) {
            snprintf(nm, sizeof(nm), "layer%02d_wk_loraA.bin", l); lw.wk_lora_a = load_bin(dir, nm, (size_t)C::LORA_RANK * C::H);
            snprintf(nm, sizeof(nm), "layer%02d_wk_loraB.bin", l); lw.wk_lora_b = load_bin(dir, nm, (size_t)C::KV_DIM * C::LORA_RANK);
        }
        if (C::LORA_V) {
            snprintf(nm, sizeof(nm), "layer%02d_wv_loraA.bin", l); lw.wv_lora_a = load_bin(dir, nm, (size_t)C::LORA_RANK * C::H);
            snprintf(nm, sizeof(nm), "layer%02d_wv_loraB.bin", l); lw.wv_lora_b = load_bin(dir, nm, (size_t)C::KV_DIM * C::LORA_RANK);
        }
        snprintf(nm, sizeof(nm), "layer%02d_o.bin", l);         w->layers[l].o         = load_bin(dir, nm, (size_t)C::H * C::Q_DIM);
        if (C::LORA_O) {
            snprintf(nm, sizeof(nm), "layer%02d_wo_loraA.bin", l); lw.wo_lora_a = load_bin(dir, nm, (size_t)C::LORA_RANK * C::Q_DIM);
            snprintf(nm, sizeof(nm), "layer%02d_wo_loraB.bin", l); lw.wo_lora_b = load_bin(dir, nm, (size_t)C::H * C::LORA_RANK);
        }
        snprintf(nm, sizeof(nm), "layer%02d_ffn_norm.bin", l);  w->layers[l].ffn_norm  = load_bin(dir, nm, (size_t)C::H);
        snprintf(nm, sizeof(nm), "layer%02d_gate_up.bin", l);   w->layers[l].gate_up   = load_bin(dir, nm, (size_t)2 * C::FFN * C::H);
        snprintf(nm, sizeof(nm), "layer%02d_down.bin", l);      w->layers[l].down      = load_bin(dir, nm, (size_t)C::H * C::FFN);
    }
}

struct Buffers {
    __half *x, *normed, *qkv, *attn, *proj, *gate_up, *ffn_mid, *logits;
    float  *scores;
    __half *k_cache, *v_cache; // [L, NKV, MAX_SEQ, HD]
    __half *lora_t;            // LoRA rank-wide scratch (A·x), only used when Config::LORA_O
    int    *d_token;
};

static void alloc_buffers(Buffers* b) {
    using C = Config;
    int lora_t_len = C::LORA_RANK > 0 ? C::LORA_RANK : 1;  // avoid a zero-size alloc
    CUDA_CHECK(cudaMalloc(&b->lora_t,  (size_t)lora_t_len * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->x,       (size_t)C::H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->normed,  (size_t)C::H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->qkv,     (size_t)C::QKV_OUT * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->attn,    (size_t)C::Q_DIM * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->proj,    (size_t)C::H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->gate_up, (size_t)2 * C::FFN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->ffn_mid, (size_t)C::FFN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->logits,  (size_t)C::VOCAB * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->scores,  (size_t)C::NH * C::MAX_SEQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->k_cache, (size_t)C::L * C::NKV * C::MAX_SEQ * C::HD * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->v_cache, (size_t)C::L * C::NKV * C::MAX_SEQ * C::HD * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->d_token, sizeof(int)));
}

// One decode step for `token` at sequence position `pos`. Updates the KV cache; computes the
// final logits only when `want_logits` (skipped during prefill of all but the last token).
static void forward(cublasHandle_t handle, const Weights* w, Buffers* b,
                    int token, int pos, bool want_logits) {
    using C = Config;
    const int H = C::H;
    const int blk = 256;

    embed_kernel<<<(H + blk - 1) / blk, blk>>>(b->x, w->embed, token, H);

    for (int l = 0; l < C::L; ++l) {
        const LayerWeights& lw = w->layers[l];
        __half* kc = b->k_cache + (size_t)l * C::NKV * C::MAX_SEQ * C::HD;
        __half* vc = b->v_cache + (size_t)l * C::NKV * C::MAX_SEQ * C::HD;

        // Attention block.
        rmsnorm_kernel<<<1, 256>>>(b->normed, b->x, lw.attn_norm, H, C::NORM_EPS);
        gemv(handle, lw.qkv, b->normed, b->qkv, H, C::QKV_OUT);
        if (C::ATTN_BIAS)
            bias_add_kernel<<<(C::QKV_OUT + blk - 1) / blk, blk>>>(b->qkv, lw.qkv_bias, C::QKV_OUT);
        // LoRA on q/k/v adds into the fused QKV slices, before RoPE (input is `normed`).
        if (C::LORA_Q) {
            lora_a_kernel<<<C::LORA_RANK, 128>>>(b->lora_t, lw.wq_lora_a, b->normed, H);
            lora_b_kernel<<<(C::Q_DIM + blk - 1) / blk, blk>>>(b->qkv, lw.wq_lora_b, b->lora_t, C::LORA_SCALE, C::Q_DIM, C::LORA_RANK);
        }
        if (C::LORA_K) {
            lora_a_kernel<<<C::LORA_RANK, 128>>>(b->lora_t, lw.wk_lora_a, b->normed, H);
            lora_b_kernel<<<(C::KV_DIM + blk - 1) / blk, blk>>>(b->qkv + C::Q_DIM, lw.wk_lora_b, b->lora_t, C::LORA_SCALE, C::KV_DIM, C::LORA_RANK);
        }
        if (C::LORA_V) {
            lora_a_kernel<<<C::LORA_RANK, 128>>>(b->lora_t, lw.wv_lora_a, b->normed, H);
            lora_b_kernel<<<(C::KV_DIM + blk - 1) / blk, blk>>>(b->qkv + C::Q_DIM + C::KV_DIM, lw.wv_lora_b, b->lora_t, C::LORA_SCALE, C::KV_DIM, C::LORA_RANK);
        }
        __half* q = b->qkv;
        __half* k = b->qkv + C::Q_DIM;
        __half* v = b->qkv + C::Q_DIM + C::KV_DIM;
        rope_qk_kernel<<<C::NH + C::NKV, 64>>>(q, k, pos);
        kv_cache_store_kernel<<<(C::NKV * C::HD + blk - 1) / blk, blk>>>(kc, vc, k, v, pos);
        attention_score_kernel<<<C::NH, 128>>>(q, kc, b->scores, pos);
        softmax_kernel<<<C::NH, 128>>>(b->scores, pos);
        attention_v_kernel<<<C::NH, 64>>>(b->attn, b->scores, vc, pos);
        gemv(handle, lw.o, b->attn, b->proj, C::Q_DIM, H);
        if (C::LORA_O) {  // proj += LORA_SCALE * B_o (A_o · attn)
            lora_a_kernel<<<C::LORA_RANK, 128>>>(b->lora_t, lw.wo_lora_a, b->attn, C::Q_DIM);
            lora_b_kernel<<<(H + blk - 1) / blk, blk>>>(b->proj, lw.wo_lora_b, b->lora_t, C::LORA_SCALE, H, C::LORA_RANK);
        }
        vadd_kernel<<<(H + blk - 1) / blk, blk>>>(b->x, b->x, b->proj, H);

        // Feed-forward block.
        rmsnorm_kernel<<<1, 256>>>(b->normed, b->x, lw.ffn_norm, H, C::NORM_EPS);
        gemv(handle, lw.gate_up, b->normed, b->gate_up, H, 2 * C::FFN);
        __half* gate = b->gate_up;
        __half* up   = b->gate_up + C::FFN;
        silu_mul_kernel<<<(C::FFN + blk - 1) / blk, blk>>>(b->ffn_mid, gate, up, C::FFN);
        gemv(handle, lw.down, b->ffn_mid, b->proj, C::FFN, H);
        vadd_kernel<<<(H + blk - 1) / blk, blk>>>(b->x, b->x, b->proj, H);
    }

    if (want_logits) {
        rmsnorm_kernel<<<1, 256>>>(b->normed, b->x, w->final_norm, H, C::NORM_EPS);
        gemv(handle, w->lm_head, b->normed, b->logits, H, C::VOCAB);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <weights_dir> <tokens.bin> <num_new_tokens>\n", argv[0]);
        return 1;
    }
    const char* dir = argv[1];
    const char* tokfile = argv[2];
    int num_new = atoi(argv[3]);
    using C = Config;

    cublasHandle_t handle;
    cublasCreate(&handle);

    Weights w;
    load_weights(&w, dir);

    Buffers b{};
    alloc_buffers(&b);

    // Read prompt tokens (raw int32).
    FILE* tf = fopen(tokfile, "rb");
    if (!tf) { fprintf(stderr, "cannot open %s\n", tokfile); return 1; }
    static int prompt[C::MAX_SEQ];
    int np = 0;
    while (np < C::MAX_SEQ && fread(&prompt[np], sizeof(int), 1, tf) == 1) np++;
    { int _extra; if (fread(&_extra, sizeof(int), 1, tf) == 1) { fprintf(stderr, "error: token file exceeds model max_seq (%d tokens) — refusing to truncate\n", C::MAX_SEQ); fclose(tf); return 1; } }
    fclose(tf);
    if (np == 0) { fprintf(stderr, "empty token file\n"); return 1; }

    // Prefill all but the last prompt token (fills KV cache; no sampling).
    int pos = 0;
    for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false);
    int cur = prompt[np - 1];

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    printf("tokens:");
    int host_idx = 0;
    for (int i = 0; i < num_new; ++i) {
        forward(handle, &w, &b, cur, pos++, true);
        argmax_kernel<<<1, 256>>>(b.d_token, b.logits, C::VOCAB);
        CUDA_CHECK(cudaMemcpy(&host_idx, b.d_token, sizeof(int), cudaMemcpyDeviceToHost));
        printf(" %d", host_idx);
        cur = host_idx;
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("\nGenerated %d tokens in %.3f s (%.1f tok/s)\n",
           num_new, ms / 1000.0f, num_new / (ms / 1000.0f));

    cublasDestroy(handle);
    return 0;
}
