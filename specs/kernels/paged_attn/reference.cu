// Plain paged GQA decode-attention — TWO-PASS reference (correctness ground truth).
//
// Computes the same exact decode-attention function as the jac-generated `paged_flash` kernel
// (online-softmax, one warp per head, gathered fp16 K/V pages via BLOCK_TABLE) but via a
// structurally DIFFERENT algorithm: materialize all scores first, a numerically-stable two-pass
// softmax over the full row, then a separate weighted-V pass. Different floating-point
// summation order by construction (mirrors windowed_attn/masked.cu's own score/softmax/value
// split) — a genuine independent cross-check, not two copies of the same fused recurrence with
// different plumbing wearing a correctness-check costume.
//
// The harness (#include's this file) already provides cuda_fp16.h / cuda_runtime.h / cmath, the
// jac_ctx struct, and the NH/HD/NKV/BLOCK_SIZE/NUM_BLOCKS/MAX_BLOCKS -D macros (all real tensor
// dims). Unlike those, LQ is a pure param, never a tensor dimension, so no harness macro covers
// it — this reference sidesteps that entirely by attending over NUM_BLOCKS*BLOCK_SIZE (the full
// page-table-covered range) instead of needing its own LQ definition; for
// shared/specs/decode_attention_paged.jac's pinned geometry these are numerically identical
// (124*32 = 3968 = LQ) by construction (the spec requires whole pages only, no partial last
// page), so this stays correct without hand-copying a magic constant that could drift from the
// spec.
#define HEADS_PER_GROUP (NH / NKV)

// phys_page(logical_page) = BLOCK_TABLE[logical_page] -- the same indirection the DSL-generated
// kernel uses (crates/jac-codegen-cuda/src/variants.rs's emit_gathered_page_step), just computed
// fresh per (head, position) here instead of cached across a loop.
__device__ __forceinline__ const __half* page_ptr(
    const __half* pages, const int* block_table, int kv_h, int s) {
    int logical_page = s / BLOCK_SIZE;
    int page_off = s % BLOCK_SIZE;
    int phys_page = block_table[logical_page];
    return pages + (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD;
}

// scores[h*L + s] = (Q[h] . K_PAGES[block_table[s/BLOCK_SIZE], kv_h, s%BLOCK_SIZE, :]) / sqrt(HD).
// Reads ALL positions (full cost, no windowing/masking -- this kernel is exact attention).
__global__ void pattn_score(
    const __half* q, const int* block_table, const __half* k_pages, float* scores) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const __half* qh = q + (size_t)h * HD;
    float scale = rsqrtf((float)HD);
    int lq = NUM_BLOCKS * BLOCK_SIZE;
    for (int s = threadIdx.x; s < lq; s += blockDim.x) {
        const __half* ks = page_ptr(k_pages, block_table, kv_h, s);
        float dot = 0.0f;
        for (int d = 0; d < HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        scores[(size_t)h * lq + s] = dot * scale;
    }
}

// In-place numerically-stable softmax over scores[h, 0..lq). One block per head.
__global__ void pattn_softmax(float* scores) {
    int lq = NUM_BLOCKS * BLOCK_SIZE;
    int h = blockIdx.x; float* row = scores + (size_t)h * lq; int tid = threadIdx.x;
    __shared__ float red[32];
    float m = -1e30f;
    for (int s = tid; s < lq; s += blockDim.x) m = fmaxf(m, row[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float mm=red[0]; for(int i=1;i<nw;++i) mm=fmaxf(mm,red[i]); red[0]=mm; }
    __syncthreads();
    float mx = red[0], sum = 0.0f;
    for (int s = tid; s < lq; s += blockDim.x) { float e = expf(row[s]-mx); row[s]=e; sum+=e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) { int nw=(blockDim.x+31)/32; float ss=0.0f; for(int i=0;i<nw;++i) ss+=red[i]; red[0]=ss; }
    __syncthreads();
    float inv = 1.0f / red[0];
    for (int s = tid; s < lq; s += blockDim.x) row[s] *= inv;
}

// O[h,d] = sum_s softmax[h,s] * V_PAGES[block_table[s/BLOCK_SIZE], kv_h, s%BLOCK_SIZE, d].
__global__ void pattn_v(
    float* out, const float* scores, const int* block_table, const __half* v_pages) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    int lq = NUM_BLOCKS * BLOCK_SIZE;
    const float* srow = scores + (size_t)h * lq;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < lq; ++s) {
            const __half* vs = page_ptr(v_pages, block_table, kv_h, s);
            acc += srow[s] * __half2float(vs[d]);
        }
        out[(size_t)h * HD + d] = acc;
    }
}

// A cuda() subject cannot define jac_workspace_bytes/jac_setup/jac_teardown: the harness already
// weak-defines them and nvcc rejects a second definition in the SAME translation unit (weak or
// not) -- mirrors windowed_attn/masked.cu's own established pattern exactly. Self-manage the
// scores[NH,lq] scratch, allocated once on the first (warmup, untimed) jac_run call. lq is
// compile-time fixed per scenario binary (NUM_BLOCKS/BLOCK_SIZE are -D-fixed per process), so
// one alloc per process is correct.
extern "C" void jac_run(jac_ctx* ctx) {
    const __half* Q = (const __half*)ctx->inputs[0];
    const int* BLOCK_TABLE = (const int*)ctx->inputs[1];
    const __half* K_PAGES = (const __half*)ctx->inputs[2];
    const __half* V_PAGES = (const __half*)ctx->inputs[3];
    float* O = (float*)ctx->outputs[0];
    static float* scores = nullptr;
    if (!scores) cudaMalloc((void**)&scores, (size_t)NH * NUM_BLOCKS * BLOCK_SIZE * sizeof(float));
    cudaStream_t st = ctx->stream;
    pattn_score   <<<NH, 128, 0, st>>>(Q, BLOCK_TABLE, K_PAGES, scores);
    pattn_softmax <<<NH, 128, 0, st>>>(scores);
    pattn_v       <<<NH,  64, 0, st>>>(O, scores, BLOCK_TABLE, V_PAGES);
}
