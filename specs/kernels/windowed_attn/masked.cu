// Windowed decode-attention — MASKED reference (track B, step 2).
//
// Computes sink+recent-window attention by reading ALL L keys/values and masking the dropped
// middle to -INF — the current model-runtime behavior (attention.rs). This is the FULL-cost
// "no-saving" baseline and the correctness reference for skip.cu.
//
// The harness (#include's this file) already provides cuda_fp16.h / cuda_runtime.h / cmath,
// the jac_ctx struct, and the NH/HD/NKV/L -D macros. The policy (S,W) is fixed below.
#ifndef SINK
#define SINK 4
#endif
#ifndef WINDOW
#define WINDOW 512
#endif
#define HEADS_PER_GROUP (NH / NKV)

// kept(s): keep the first SINK tokens + the last WINDOW tokens of an L-long cache.
__device__ __forceinline__ bool attn_kept(int s) { return (s < SINK) || (s >= L - WINDOW); }

// scores[h*L + s] = masked (Q[h] . K[kv_h,s]) / sqrt(HD).  Reads ALL K (full cost).
__global__ void wattn_score(const __half* q, const __half* k, float* scores) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const __half* qh = q + (size_t)h * HD;
    const __half* kbase = k + (size_t)kv_h * L * HD;
    float scale = rsqrtf((float)HD);
    for (int s = threadIdx.x; s < L; s += blockDim.x) {
        const __half* ks = kbase + (size_t)s * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        float sv = dot * scale;
        if (!attn_kept(s)) sv = -INFINITY;            // mask, but K already read (no saving)
        scores[(size_t)h * L + s] = sv;
    }
}

// In-place numerically-stable softmax over scores[h, 0..L). One block per head.
__global__ void wattn_softmax(float* scores) {
    int h = blockIdx.x; float* row = scores + (size_t)h * L; int tid = threadIdx.x;
    __shared__ float red[32];
    float m = -1e30f;
    for (int s = tid; s < L; s += blockDim.x) m = fmaxf(m, row[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float mm=red[0]; for(int i=1;i<nw;++i) mm=fmaxf(mm,red[i]); red[0]=mm; }
    __syncthreads();
    float mx = red[0], sum = 0.0f;
    for (int s = tid; s < L; s += blockDim.x) { float e = expf(row[s]-mx); row[s]=e; sum+=e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float ss=0.0f; for(int i=0;i<nw;++i) ss+=red[i]; red[0]=ss; }
    __syncthreads();
    float inv = 1.0f / red[0];
    for (int s = tid; s < L; s += blockDim.x) row[s] *= inv;
}

// O[h,d] = sum_s softmax[h,s] * V[kv_h,s,d].  Reads ALL V (full cost). fp32 output.
__global__ void wattn_v(float* out, const float* scores, const __half* v) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const float* srow = scores + (size_t)h * L;
    const __half* vbase = v + (size_t)kv_h * L * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < L; ++s)
            acc += srow[s] * __half2float(vbase[(size_t)s * HD + d]);   // V read for every s
        out[(size_t)h * HD + d] = acc;
    }
}

// A cuda() subject cannot define jac_workspace_bytes/jac_setup/jac_teardown: the harness already
// weak-defines them and nvcc rejects a second definition in the SAME translation unit (weak or
// not). So we self-manage the scores[NH,L] scratch — allocated once on the first (warmup, untimed)
// jac_run call. L is compile-time fixed per scenario binary, so one alloc per process is correct.
extern "C" void jac_run(jac_ctx* ctx) {
    const __half* Q = (const __half*)ctx->inputs[0];
    const __half* K = (const __half*)ctx->inputs[1];
    const __half* V = (const __half*)ctx->inputs[2];
    float* O = (float*)ctx->outputs[0];
    static float* scores = nullptr;
    if (!scores) cudaMalloc((void**)&scores, (size_t)NH * L * sizeof(float));
    cudaStream_t st = ctx->stream;
    wattn_score   <<<NH, 128, 0, st>>>(Q, K, scores);
    wattn_softmax <<<NH, 128, 0, st>>>(scores);
    wattn_v       <<<NH,  64, 0, st>>>(O, scores, V);
}
