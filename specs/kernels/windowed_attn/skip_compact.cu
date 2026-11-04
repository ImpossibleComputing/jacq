// Windowed decode-attention — COMPACTED skip variant (track B, step 2).
//
// Like skip.cu, but it does not even iterate the dropped slots: it processes exactly the
// `kept = min(L, SINK+WINDOW)` kept keys, mapping compacted slot j -> original key index via the
// two contiguous ranges [0,SINK) (sink) and [L-WINDOW, L) (recent window). This removes skip.cu's
// O(L) floor (skip still WRITES all L score slots and softmaxes over L); compacted is O(kept)
// end-to-end. Output is still bitwise-identical to masked: the kept keys are visited in the same
// ascending order (sinks, then window), so the score/softmax/weighted-V float ops match exactly.
//
// Edge case L <= SINK+WINDOW (the window covers the whole context, incl. the sink): the two ranges
// would overlap and double-count, so we fall back to the identity map over all L keys (kept = L).
#ifndef SINK
#define SINK 4
#endif
#ifndef WINDOW
#define WINDOW 512
#endif
#define HEADS_PER_GROUP (NH / NKV)

// Number of distinct kept keys, and the compacted-slot -> original-key map. When the window covers
// the context (L <= SINK+WINDOW) every key is kept and the map is the identity.
__device__ __forceinline__ int attn_kept_count() { return (L <= SINK + WINDOW) ? L : (SINK + WINDOW); }
__device__ __forceinline__ int attn_orig(int j) {
    return (L <= SINK + WINDOW) ? j : ((j < SINK) ? j : (L - WINDOW + (j - SINK)));
}

// scores[h*kept + j] = (Q[h] . K[kv_h, orig(j)]) / sqrt(HD), for the kept keys only.
__global__ void wattn_score(const __half* q, const __half* k, float* scores) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const __half* qh = q + (size_t)h * HD;
    const __half* kbase = k + (size_t)kv_h * L * HD;
    float scale = rsqrtf((float)HD);
    int kept = attn_kept_count();
    for (int j = threadIdx.x; j < kept; j += blockDim.x) {
        int s = attn_orig(j);
        const __half* ks = kbase + (size_t)s * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        scores[(size_t)h * kept + j] = dot * scale;
    }
}

// In-place numerically-stable softmax over the kept slots scores[h, 0..kept). One block per head.
__global__ void wattn_softmax(float* scores) {
    int h = blockIdx.x; int kept = attn_kept_count();
    float* row = scores + (size_t)h * kept; int tid = threadIdx.x;
    __shared__ float red[32];
    float m = -1e30f;
    for (int s = tid; s < kept; s += blockDim.x) m = fmaxf(m, row[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float mm=red[0]; for(int i=1;i<nw;++i) mm=fmaxf(mm,red[i]); red[0]=mm; }
    __syncthreads();
    float mx = red[0], sum = 0.0f;
    for (int s = tid; s < kept; s += blockDim.x) { float e = expf(row[s]-mx); row[s]=e; sum+=e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float ss=0.0f; for(int i=0;i<nw;++i) ss+=red[i]; red[0]=ss; }
    __syncthreads();
    float inv = 1.0f / red[0];
    for (int s = tid; s < kept; s += blockDim.x) row[s] *= inv;
}

// O[h,d] = sum over kept slots j of softmax[h,j] * V[kv_h, orig(j), d]. fp32 output.
__global__ void wattn_v(float* out, const float* scores, const __half* v) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    int kept = attn_kept_count();
    const float* srow = scores + (size_t)h * kept;
    const __half* vbase = v + (size_t)kv_h * L * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < kept; ++j)
            acc += srow[j] * __half2float(vbase[(size_t)attn_orig(j) * HD + d]);
        out[(size_t)h * HD + d] = acc;
    }
}

// Self-managed scratch (see masked.cu): scores[NH, kept]. kept is compile-time fixed per scenario.
extern "C" void jac_run(jac_ctx* ctx) {
    const __half* Q = (const __half*)ctx->inputs[0];
    const __half* K = (const __half*)ctx->inputs[1];
    const __half* V = (const __half*)ctx->inputs[2];
    float* O = (float*)ctx->outputs[0];
    static float* scores = nullptr;
    int kept = (L <= SINK + WINDOW) ? L : (SINK + WINDOW);
    if (!scores) cudaMalloc((void**)&scores, (size_t)NH * kept * sizeof(float));
    cudaStream_t st = ctx->stream;
    wattn_score   <<<NH, 128, 0, st>>>(Q, K, scores);
    wattn_softmax <<<NH, 128, 0, st>>>(scores);
    wattn_v       <<<NH,  64, 0, st>>>(O, scores, V);
}
