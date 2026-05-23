import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST   = 0x14F0000000000000ULL;
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000ULL;

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__
half2 silu_mul_h2(float x0, float x1, float y0, float y1) {
    const float2 x = make_float2(x0, x1);
    const float2 y = make_float2(y0, y1);
    const float2 e = make_float2(
        __expf(-x.x),
        __expf(-x.y)
    );
    const float2 s = make_float2(
        __fdividef(x.x, 1.0f + e.x),
        __fdividef(x.y, 1.0f + e.y)
    );
    const float2 p = __fmul2_rn(s, y);
    return __float22half2_rn(p);
}

__device__ __forceinline__ uint32_t bitcast_u32(half2 h) {
    union {
        half2 h;
        uint32_t u;
    } x;
    x.h = h;
    return x.u;
}

__device__ __forceinline__
void stg_32b(const void* dst, unsigned long long v0, unsigned long long v1,
            unsigned long long v2, unsigned long long v3) {
    asm volatile(
        "st.global.wt.v4.b64 [%0], {%1, %2, %3, %4};"
        :: "l"(dst), "l"(v0), "l"(v1), "l"(v2), "l"(v3)
        : "memory"
    );
}

__device__ __forceinline__
uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

__device__ __forceinline__
void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ __forceinline__
void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAX:\n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_RELAX;\n\t"
        "bra.uni LAB_WAIT_RELAX;\n\t"
        "DONE_RELAX:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

template <int CTA_GROUP>
__device__ __forceinline__
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %7;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr), "n"(CTA_GROUP), "l"(cache_policy)
        : "memory"
    );
}

template <int CTA_GROUP>
__device__ __forceinline__
void tma_3d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z,
                           int mbar_addr, uint16_t cta_mask, uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::%6.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %7, %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr), "n"(CTA_GROUP), "h"(cta_mask), "l"(cache_policy)
        : "memory"
    );
}

__device__ __forceinline__
void tcgen05_cp_cta2(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__
void tcgen05_mma_cta2(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

__device__ __forceinline__
void tcgen05_mma_cta2_collector_fill(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

__device__ __forceinline__
void tcgen05_mma_cta2_collector_lastuse(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

__device__ __forceinline__
void tcgen05_ld_32x32bx8(float *tmp, int addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr)
    );
}

// ---------------- TensorMap creation ----------------

void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *error_msg_ptr;
    if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
        error_msg_ptr = "unable to get error string";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void init_AB_tmap(
    CUtensorMap *tmap,
    const char *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width
) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
    uint64_t globalStrides[rank-1] = {global_width / 2, 128};
    uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    auto err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

void init_SF_tmap(
    CUtensorMap *tmap,
    const char *ptr,
    uint64_t mn,
    uint64_t K,
    uint32_t block_k
) {
    constexpr uint32_t rank = 3;
    const uint64_t k_blocks = K / 64;
    const uint64_t mn_blocks = mn / 128;
    const uint32_t tile_k_blocks = block_k / 64;
    constexpr uint64_t SF_BLOCK_BYTES = 512;
    constexpr uint64_t X_ELEMS = SF_BLOCK_BYTES / sizeof(uint16_t);
    uint64_t globalDim[rank]       = {X_ELEMS, mn_blocks, k_blocks};
    uint64_t globalStrides[rank-1] = {k_blocks * SF_BLOCK_BYTES, SF_BLOCK_BYTES};
    uint32_t boxDim[rank]          = {(uint32_t)X_ELEMS, 1, tile_k_blocks};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    auto err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

// Cache cuTensorMapEncodeTiled() results keyed by (ptr + shape/tile params).
struct ABKey {
    uint64_t ptr;
    uint32_t dev;
    uint32_t global_h;
    uint32_t global_w;
    uint32_t shared_h;
    uint32_t shared_w;
};

struct SFKey {
    uint64_t ptr;
    uint32_t dev;
    uint32_t mn;
    uint32_t K;
    uint32_t block_k;
};

static inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
    return h;
}

struct ABKeyHash {
    size_t operator()(ABKey const& k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        h = fnv1a_u64(h, k.ptr);
        h = fnv1a_u64(h, uint64_t(k.dev));
        h = fnv1a_u64(h, (uint64_t(k.global_h) << 32) | uint64_t(k.global_w));
        h = fnv1a_u64(h, (uint64_t(k.shared_h) << 32) | uint64_t(k.shared_w));
        return (size_t)h;
    }
};
struct ABKeyEq {
    bool operator()(ABKey const& a, ABKey const& b) const noexcept {
        return a.ptr == b.ptr
            && a.dev == b.dev
            && a.global_h == b.global_h
            && a.global_w == b.global_w
            && a.shared_h == b.shared_h
            && a.shared_w == b.shared_w;
    }
};

struct SFKeyHash {
    size_t operator()(SFKey const& k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        h = fnv1a_u64(h, k.ptr);
        h = fnv1a_u64(h, uint64_t(k.dev));
        h = fnv1a_u64(h, (uint64_t(k.mn) << 32) | uint64_t(k.K));
        h = fnv1a_u64(h, uint64_t(k.block_k));
        return (size_t)h;
    }
};
struct SFKeyEq {
    bool operator()(SFKey const& a, SFKey const& b) const noexcept {
        return a.ptr == b.ptr && a.dev == b.dev && a.mn == b.mn && a.K == b.K && a.block_k == b.block_k;
    }
};

static std::unordered_map<ABKey, CUtensorMap, ABKeyHash, ABKeyEq> g_ab_cache;
static std::unordered_map<SFKey, CUtensorMap, SFKeyHash, SFKeyEq> g_sf_cache;

static std::once_flag g_tmap_cache_once;
static inline void init_tmap_caches_once() {
    // Keep this large enough to avoid rehashing for typical workloads (few shapes).
    // Reduces first-hit latency spikes and keeps last-hit pointers stable.
    g_ab_cache.max_load_factor(0.7f);
    g_sf_cache.max_load_factor(0.7f);
    g_ab_cache.reserve(256);
    g_sf_cache.reserve(256);
}

static inline const CUtensorMap& get_ab_tmap_cached(
    const char* ptr, uint32_t dev, uint32_t global_h, uint32_t global_w, uint32_t shared_h, uint32_t shared_w
) {
    std::call_once(g_tmap_cache_once, init_tmap_caches_once);

    // Thread-local last-hit fast path (common in steady-state loops).
    static thread_local ABKey last_key{};
    static thread_local const CUtensorMap* last_val = nullptr;
    static thread_local bool last_valid = false;

    ABKey key{(uint64_t)ptr, dev, global_h, global_w, shared_h, shared_w};
    if (last_valid && ABKeyEq{}(key, last_key)) return *last_val;
    auto it = g_ab_cache.find(key);
    if (it != g_ab_cache.end()) {
        last_key = key;
        last_val = &it->second;
        last_valid = true;
        return it->second;
    }
    CUtensorMap tmp{};
    init_AB_tmap(&tmp, ptr, global_h, global_w, shared_h, shared_w);
    auto ins = g_ab_cache.emplace(key, tmp);
    last_key = key;
    last_val = &ins.first->second;
    last_valid = true;
    return ins.first->second;
}

static inline const CUtensorMap& get_sf_tmap_cached(
    const char* ptr, uint32_t dev, uint32_t mn, uint32_t K, uint32_t block_k
) {
    std::call_once(g_tmap_cache_once, init_tmap_caches_once);

    static thread_local SFKey last_key{};
    static thread_local const CUtensorMap* last_val = nullptr;
    static thread_local bool last_valid = false;

    SFKey key{(uint64_t)ptr, dev, mn, K, block_k};
    if (last_valid && SFKeyEq{}(key, last_key)) return *last_val;
    auto it = g_sf_cache.find(key);
    if (it != g_sf_cache.end()) {
        last_key = key;
        last_val = &it->second;
        last_valid = true;
        return it->second;
    }
    CUtensorMap tmp{};
    init_SF_tmap(&tmp, ptr, mn, K, block_k);
    auto ins = g_sf_cache.emplace(key, tmp);
    last_key = key;
    last_val = &ins.first->second;
    last_valid = true;
    return ins.first->second;
}

// ============================================================================
// N=64 collector kernel (M=256 path)
// ============================================================================
template <int BLOCK_M, int BLOCK_K, int NUM_STAGES>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_cta2_collector_n64_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap,
    const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr,
    int M, int N, int K
) {
    constexpr int CTA_GROUP = 2;
    constexpr int BLOCK_N = 64;
    constexpr int HALF_BLOCK_N = BLOCK_N / CTA_GROUP;
    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_pid = bid / CTA_GROUP;
    const int grid_n_clusters = N / BLOCK_N;
    const int cluster_m = cluster_pid / grid_n_clusters;
    const int cluster_n = cluster_pid % grid_n_clusters;
    const int off_m = cluster_m * (BLOCK_M * CTA_GROUP) + cta_rank * BLOCK_M;
    const int off_n = cluster_n * BLOCK_N;
    const int sf_y_A = off_m / 128;
    const int sf_y_B = off_n / 128;
    const int B_col_offset = off_n + cta_rank * HALF_BLOCK_N;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int A_size    = BLOCK_M * BLOCK_K / 2;
    constexpr int B1_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int B2_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int SFA_size  = 128 * BLOCK_K / 16;
    constexpr int SFB1_size = 128 * BLOCK_K / 16;
    constexpr int SFB2_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B1_size + B2_size + SFA_size + SFB1_size + SFB2_size;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    constexpr int ACC_BASE = 0;
    constexpr int ACC1_OFF = 0;
    constexpr int ACC2_OFF = BLOCK_N;
    constexpr int SFA_COLS_PER_K = 8;
    constexpr int SFB_COLS_PER_K = 4;
    constexpr int SFA_tmem  = 2 * BLOCK_N;
    constexpr int SFB1_tmem = SFA_tmem + SFA_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int SFB2_tmem = SFB1_tmem + SFB_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int TOTAL_TMEM_COLS = 256;

    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        mbarrier_init(mainloop_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(addr), "r"(TOTAL_TMEM_COLS));
    }
    asm volatile("bar.sync 1, %0;" :: "r"(32) : "memory");

    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | (2U << 27U);
    constexpr int SBO_AB = 8 * 128;
    constexpr int SBO_SF = 8 * 16;
    constexpr uint64_t AB_desc_base = (desc_encode(SBO_AB) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    constexpr uint64_t SF_desc_base = (desc_encode(SBO_SF) << 32ULL) | (1ULL << 46ULL);

    const int num_iters = K / BLOCK_K;

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        // Mirror the heuristic used in `single_gemm_ref.py`.
        const uint64_t cache_A = EVICT_FIRST;
        const uint64_t cache_B = EVICT_FIRST;
        int tma_stage = 0;
        int mma_phase = 1;
        int it = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++, it++) {
            if (it >= NUM_STAGES)
                mbarrier_wait_relaxed(mma_mbar_addr + tma_stage * 8, mma_phase);

            const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size;
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = base_smem + A_size + B1_size + B2_size;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            constexpr int TENSOR_TMA_SIZE = A_size + B1_size + B2_size;
            const int SF_TMA_SIZE = SFA_size + ((cta_rank == 0) ? (CTA_GROUP * (SFB1_size + SFB2_size)) : 0);
            const int TOTAL_TMA_SIZE = TENSOR_TMA_SIZE + SF_TMA_SIZE;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(mbar_addr), "r"(TOTAL_TMA_SIZE) : "memory");

            const int z_ab = iter_k * (BLOCK_K / 256);
            const int z_sf = iter_k * (BLOCK_K / 64);

            // (kept as-is from current submission_13901.py)
            tma_3d_gmem2smem<CTA_GROUP>(B2_smem, &B2_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(B1_smem, &B1_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, z_ab, mbar_addr, cache_A);
            if (cta_rank == 0) {
                constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB2_smem, &SFB2_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB1_smem, &SFB1_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
            }
            tma_3d_gmem2smem<CTA_GROUP>(SFA_smem, &SFA_tmap, 0, sf_y_A, z_sf, mbar_addr, cache_A);

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) mma_phase ^= 1;
        }
    } else if (cta_rank == 0 && warp_id == NUM_WARPS - 1 && elect_sync()) {
        int tma_stage = 0;
        int tma_phase = 0;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        const int scale_B_base_off = (cluster_n & 1) * (BLOCK_N / 32);

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            // Doc-aligned pattern for pipelined tcgen ops across threads:
            // wait (relaxed) -> tcgen05.fence::after_thread_sync -> tcgen ops
            mbarrier_wait_relaxed(tma_mbar_addr + tma_stage * 8, tma_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size;
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = base_smem + A_size + B1_size + B2_size;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            const uint64_t SFA_desc  = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB1_desc = SF_desc_base + ((uint64_t)SFB1_smem >> 4ULL);
            const uint64_t SFB2_desc = SF_desc_base + ((uint64_t)SFB2_smem >> 4ULL);

            constexpr int SF_ITERS = BLOCK_K / MMA_K;
            constexpr int MMA_ITERS = BLOCK_K / MMA_K;
            constexpr int HALF = (SF_ITERS > 1) ? (SF_ITERS / 2) : 1;

            uint64_t a_descs[MMA_ITERS];
            uint64_t b1_descs[MMA_ITERS];
            uint64_t b2_descs[MMA_ITERS];
            #pragma unroll
            for (int k2 = 0; k2 < MMA_ITERS; k2++) {
                const int off = k2 * 32;
                a_descs[k2] = AB_desc_base + desc_encode(A_smem + off);
                b1_descs[k2] = AB_desc_base + desc_encode(B1_smem + off);
                b2_descs[k2] = AB_desc_base + desc_encode(B2_smem + off);
            }
            const int scale_A_base = SFA_tmem;
            const int scale_B1_base = SFB1_tmem + scale_B_base_off;
            const int scale_B2_base = SFB2_tmem + scale_B_base_off;

            // (kept as-is from current submission_13901.py)
            #pragma unroll
            for (int k = 0; k < HALF; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = 0; k2 < HALF; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = (k2 == 0) ? iter_k : 1;

                tcgen05_mma_cta2_collector_fill(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2_collector_lastuse(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            #pragma unroll
            for (int k = HALF; k < SF_ITERS; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = HALF; k2 < MMA_ITERS; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = 1;

                tcgen05_mma_cta2_collector_fill(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2_collector_lastuse(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            // Ensure all tcgen ops are ordered before signaling stage completion.
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_addr + tma_stage * 8), "h"(cta_mask) : "memory");

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) tma_phase ^= 1;
        }
        // Ensure all tcgen ops are ordered before signaling mainloop completion.
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mainloop_mbar_addr), "h"(cta_mask) : "memory");
    } else if (warp_id < 4) {
        // Relaxed mainloop completion wait: lower sync overhead.
        mbarrier_wait_relaxed(mainloop_mbar_addr, 0);
        // Restore tcgen fence for TMEM visibility ordering after relaxed wait.
        asm volatile("tcgen05.fence::after_thread_sync;");
        const int taddr = tmem_addr[0];

        if (tid < BLOCK_M) {
            constexpr int WIDTH = 64;
            const int tmem_row = cta_rank * 128 + warp_id * 32;
            half* row_ptr = C_ptr + (off_m + tid) * N + off_n;
            const int row_base1 = taddr + (tmem_row << 16) + (ACC_BASE + ACC1_OFF);
            const int row_base2 = taddr + (tmem_row << 16) + (ACC_BASE + ACC2_OFF);

            // Epilogue pipelined: overlap tcgen05.ld for next chunk with compute/store.
            float acc1a[16], acc2a[16];
            float acc1b[16], acc2b[16];

            // Base 0 -> A
            tcgen05_ld_32x32bx8(acc1a + 0, row_base1 + 0);
            tcgen05_ld_32x32bx8(acc1a + 8, row_base1 + 8);
            tcgen05_ld_32x32bx8(acc2a + 0, row_base2 + 0);
            tcgen05_ld_32x32bx8(acc2a + 8, row_base2 + 8);

            // Base 0
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            // Prefetch base 16 -> B
            tcgen05_ld_32x32bx8(acc1b + 0, row_base1 + 16);
            tcgen05_ld_32x32bx8(acc1b + 8, row_base1 + 24);
            tcgen05_ld_32x32bx8(acc2b + 0, row_base2 + 16);
            tcgen05_ld_32x32bx8(acc2b + 8, row_base2 + 24);

            {
                half2 h0 = silu_mul_h2(acc1a[0],  acc1a[1],  acc2a[0],  acc2a[1]);
                half2 h1 = silu_mul_h2(acc1a[2],  acc1a[3],  acc2a[2],  acc2a[3]);
                half2 h2 = silu_mul_h2(acc1a[4],  acc1a[5],  acc2a[4],  acc2a[5]);
                half2 h3 = silu_mul_h2(acc1a[6],  acc1a[7],  acc2a[6],  acc2a[7]);
                half2 h4 = silu_mul_h2(acc1a[8],  acc1a[9],  acc2a[8],  acc2a[9]);
                half2 h5 = silu_mul_h2(acc1a[10], acc1a[11], acc2a[10], acc2a[11]);
                half2 h6 = silu_mul_h2(acc1a[12], acc1a[13], acc2a[12], acc2a[13]);
                half2 h7 = silu_mul_h2(acc1a[14], acc1a[15], acc2a[14], acc2a[15]);

                const uint32_t u0 = bitcast_u32(h0);
                const uint32_t u1 = bitcast_u32(h1);
                const uint32_t u2 = bitcast_u32(h2);
                const uint32_t u3 = bitcast_u32(h3);
                const uint32_t u4 = bitcast_u32(h4);
                const uint32_t u5 = bitcast_u32(h5);
                const uint32_t u6 = bitcast_u32(h6);
                const uint32_t u7 = bitcast_u32(h7);

                const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                stg_32b((const void*)(row_ptr + 0), q0, q1, q2, q3);
            }

            // Base 16
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            // Prefetch base 32 -> A
            tcgen05_ld_32x32bx8(acc1a + 0, row_base1 + 32);
            tcgen05_ld_32x32bx8(acc1a + 8, row_base1 + 40);
            tcgen05_ld_32x32bx8(acc2a + 0, row_base2 + 32);
            tcgen05_ld_32x32bx8(acc2a + 8, row_base2 + 40);
            {
                half2 h0 = silu_mul_h2(acc1b[0],  acc1b[1],  acc2b[0],  acc2b[1]);
                half2 h1 = silu_mul_h2(acc1b[2],  acc1b[3],  acc2b[2],  acc2b[3]);
                half2 h2 = silu_mul_h2(acc1b[4],  acc1b[5],  acc2b[4],  acc2b[5]);
                half2 h3 = silu_mul_h2(acc1b[6],  acc1b[7],  acc2b[6],  acc2b[7]);
                half2 h4 = silu_mul_h2(acc1b[8],  acc1b[9],  acc2b[8],  acc2b[9]);
                half2 h5 = silu_mul_h2(acc1b[10], acc1b[11], acc2b[10], acc2b[11]);
                half2 h6 = silu_mul_h2(acc1b[12], acc1b[13], acc2b[12], acc2b[13]);
                half2 h7 = silu_mul_h2(acc1b[14], acc1b[15], acc2b[14], acc2b[15]);

                const uint32_t u0 = bitcast_u32(h0);
                const uint32_t u1 = bitcast_u32(h1);
                const uint32_t u2 = bitcast_u32(h2);
                const uint32_t u3 = bitcast_u32(h3);
                const uint32_t u4 = bitcast_u32(h4);
                const uint32_t u5 = bitcast_u32(h5);
                const uint32_t u6 = bitcast_u32(h6);
                const uint32_t u7 = bitcast_u32(h7);

                const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                stg_32b((const void*)(row_ptr + 16), q0, q1, q2, q3);
            }

            // Base 32
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            // Prefetch base 48 -> B
            tcgen05_ld_32x32bx8(acc1b + 0, row_base1 + 48);
            tcgen05_ld_32x32bx8(acc1b + 8, row_base1 + 56);
            tcgen05_ld_32x32bx8(acc2b + 0, row_base2 + 48);
            tcgen05_ld_32x32bx8(acc2b + 8, row_base2 + 56);
            {
                half2 h0 = silu_mul_h2(acc1a[0],  acc1a[1],  acc2a[0],  acc2a[1]);
                half2 h1 = silu_mul_h2(acc1a[2],  acc1a[3],  acc2a[2],  acc2a[3]);
                half2 h2 = silu_mul_h2(acc1a[4],  acc1a[5],  acc2a[4],  acc2a[5]);
                half2 h3 = silu_mul_h2(acc1a[6],  acc1a[7],  acc2a[6],  acc2a[7]);
                half2 h4 = silu_mul_h2(acc1a[8],  acc1a[9],  acc2a[8],  acc2a[9]);
                half2 h5 = silu_mul_h2(acc1a[10], acc1a[11], acc2a[10], acc2a[11]);
                half2 h6 = silu_mul_h2(acc1a[12], acc1a[13], acc2a[12], acc2a[13]);
                half2 h7 = silu_mul_h2(acc1a[14], acc1a[15], acc2a[14], acc2a[15]);

                const uint32_t u0 = bitcast_u32(h0);
                const uint32_t u1 = bitcast_u32(h1);
                const uint32_t u2 = bitcast_u32(h2);
                const uint32_t u3 = bitcast_u32(h3);
                const uint32_t u4 = bitcast_u32(h4);
                const uint32_t u5 = bitcast_u32(h5);
                const uint32_t u6 = bitcast_u32(h6);
                const uint32_t u7 = bitcast_u32(h7);

                const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                stg_32b((const void*)(row_ptr + 32), q0, q1, q2, q3);
            }

            // Base 48
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            {
                half2 h0 = silu_mul_h2(acc1b[0],  acc1b[1],  acc2b[0],  acc2b[1]);
                half2 h1 = silu_mul_h2(acc1b[2],  acc1b[3],  acc2b[2],  acc2b[3]);
                half2 h2 = silu_mul_h2(acc1b[4],  acc1b[5],  acc2b[4],  acc2b[5]);
                half2 h3 = silu_mul_h2(acc1b[6],  acc1b[7],  acc2b[6],  acc2b[7]);
                half2 h4 = silu_mul_h2(acc1b[8],  acc1b[9],  acc2b[8],  acc2b[9]);
                half2 h5 = silu_mul_h2(acc1b[10], acc1b[11], acc2b[10], acc2b[11]);
                half2 h6 = silu_mul_h2(acc1b[12], acc1b[13], acc2b[12], acc2b[13]);
                half2 h7 = silu_mul_h2(acc1b[14], acc1b[15], acc2b[14], acc2b[15]);

                const uint32_t u0 = bitcast_u32(h0);
                const uint32_t u1 = bitcast_u32(h1);
                const uint32_t u2 = bitcast_u32(h2);
                const uint32_t u3 = bitcast_u32(h3);
                const uint32_t u4 = bitcast_u32(h4);
                const uint32_t u5 = bitcast_u32(h5);
                const uint32_t u6 = bitcast_u32(h6);
                const uint32_t u7 = bitcast_u32(h7);

                const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                stg_32b((const void*)(row_ptr + 48), q0, q1, q2, q3);
            }
        asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");    
        if (warp_id == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TOTAL_TMEM_COLS));                
        }
    }

}

// ============================================================================
// N=128 kernel WITHOUT collector (M=512 path)
// ============================================================================
template <int BLOCK_M, int BLOCK_K, int NUM_STAGES>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_cta2_baseline_n128_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap,
    const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr,
    int M, int N, int K
) {
    constexpr int CTA_GROUP = 2;
    constexpr int BLOCK_N = 128;
    constexpr int HALF_BLOCK_N = BLOCK_N / CTA_GROUP;
    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_pid = bid / CTA_GROUP;
    const int grid_n_clusters = N / BLOCK_N;
    const int cluster_m = cluster_pid / grid_n_clusters;
    const int cluster_n = cluster_pid % grid_n_clusters;
    const int off_m = cluster_m * (BLOCK_M * CTA_GROUP) + cta_rank * BLOCK_M;
    const int off_n = cluster_n * BLOCK_N;
    const int sf_y_A = off_m / 128;
    const int sf_y_B = off_n / 128;
    const int B_col_offset = off_n + cta_rank * HALF_BLOCK_N;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int A_size    = BLOCK_M * BLOCK_K / 2;
    constexpr int B1_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int B2_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int SFA_size  = 128 * BLOCK_K / 16;
    constexpr int SFB1_size = 128 * BLOCK_K / 16;
    constexpr int SFB2_size = 128 * BLOCK_K / 16;
    constexpr int PAD = 128;
    constexpr int STAGE_SIZE = A_size + PAD + B1_size + B2_size + PAD + SFA_size + SFB1_size + SFB2_size;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    constexpr int ACC_BASE = 0;
    constexpr int ACC1_OFF = 0;
    constexpr int ACC2_OFF = BLOCK_N;
    constexpr int SFA_COLS_PER_K = 8;
    constexpr int SFB_COLS_PER_K = 4;
    constexpr int SFA_tmem  = 2 * BLOCK_N;
    constexpr int SFB1_tmem = SFA_tmem + SFA_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int SFB2_tmem = SFB1_tmem + SFB_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int TOTAL_TMEM_COLS = 512;

    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        mbarrier_init(mainloop_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(addr), "r"(TOTAL_TMEM_COLS));
    }

    asm volatile("bar.sync 1, %0;" :: "r"(32) : "memory");

    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | (2U << 27U);
    constexpr int SBO_AB = 8 * 128;
    constexpr int SBO_SF = 8 * 16;
    constexpr uint64_t AB_desc_base = (desc_encode(SBO_AB) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    constexpr uint64_t SF_desc_base = (desc_encode(SBO_SF) << 32ULL) | (1ULL << 46ULL);

    const int num_iters = K / BLOCK_K;

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        // Mirror the heuristic used in `single_gemm_ref.py`.
        const uint64_t cache_A = EVICT_FIRST;
        const uint64_t cache_B = EVICT_FIRST;
        int tma_stage = 0;
        int mma_phase = 1;
        int it = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++, it++) {
            if (it >= NUM_STAGES)
                mbarrier_wait_relaxed(mma_mbar_addr + tma_stage * 8, mma_phase);

            const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size + PAD;
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = B2_smem + B2_size + PAD;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            constexpr int TENSOR_TMA_SIZE = A_size + B1_size + B2_size;
            const int SF_TMA_SIZE = SFA_size + ((cta_rank == 0) ? (CTA_GROUP * (SFB1_size + SFB2_size)) : 0);
            const int TOTAL_TMA_SIZE = TENSOR_TMA_SIZE + SF_TMA_SIZE;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(mbar_addr), "r"(TOTAL_TMA_SIZE) : "memory");

            const int z_ab = iter_k * (BLOCK_K / 256);
            const int z_sf = iter_k * (BLOCK_K / 64);
            tma_3d_gmem2smem<CTA_GROUP>(B1_smem, &B1_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(B2_smem, &B2_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, z_ab, mbar_addr, cache_A);
            if (cta_rank == 0) {
                constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB1_smem, &SFB1_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB2_smem, &SFB2_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
            }
            tma_3d_gmem2smem<CTA_GROUP>(SFA_smem, &SFA_tmap, 0, sf_y_A, z_sf, mbar_addr, cache_A);


            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) mma_phase ^= 1;
        }
    } else if (cta_rank == 0 && warp_id == NUM_WARPS - 1 && elect_sync()) {
        int tma_stage = 0;
        int tma_phase = 0;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        constexpr int scale_B_base_off = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            // Doc-aligned pattern for pipelined tcgen ops across threads:
            // wait (relaxed) -> tcgen05.fence::after_thread_sync -> tcgen ops
            mbarrier_wait_relaxed(tma_mbar_addr + tma_stage * 8, tma_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size + PAD;
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = B2_smem + B2_size + PAD;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            const uint64_t SFA_desc  = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB1_desc = SF_desc_base + ((uint64_t)SFB1_smem >> 4ULL);
            const uint64_t SFB2_desc = SF_desc_base + ((uint64_t)SFB2_smem >> 4ULL);

            constexpr int SF_ITERS = BLOCK_K / MMA_K;
            constexpr int MMA_ITERS = BLOCK_K / MMA_K;
            constexpr int HALF = (SF_ITERS > 1) ? (SF_ITERS / 2) : 1;
            uint64_t a_descs[MMA_ITERS];
            uint64_t b1_descs[MMA_ITERS];
            uint64_t b2_descs[MMA_ITERS];
            #pragma unroll
            for (int k2 = 0; k2 < MMA_ITERS; k2++) {
                const int off = k2 * 32;
                a_descs[k2] = AB_desc_base + desc_encode(A_smem + off);
                b1_descs[k2] = AB_desc_base + desc_encode(B1_smem + off);
                b2_descs[k2] = AB_desc_base + desc_encode(B2_smem + off);
            }
            const int scale_A_base = SFA_tmem;
            const int scale_B1_base = SFB1_tmem + scale_B_base_off;
            const int scale_B2_base = SFB2_tmem + scale_B_base_off;

            #pragma unroll
            for (int k = 0; k < HALF; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = 0; k2 < HALF; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = (k2 == 0) ? iter_k : 1;
                tcgen05_mma_cta2(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            #pragma unroll
            for (int k = HALF; k < SF_ITERS; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = HALF; k2 < MMA_ITERS; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = 1;
                tcgen05_mma_cta2(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            // Ensure all tcgen ops are ordered before signaling stage completion.
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_addr + tma_stage * 8), "h"(cta_mask) : "memory");

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) tma_phase ^= 1;
        }
        // Ensure all tcgen ops are ordered before signaling mainloop completion.
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mainloop_mbar_addr), "h"(cta_mask) : "memory");
    } else if (warp_id < 4) {
        // Relaxed mainloop completion wait: lower sync overhead.
        mbarrier_wait_relaxed(mainloop_mbar_addr, 0);
        // Restore tcgen fence for TMEM visibility ordering after relaxed wait.
        asm volatile("tcgen05.fence::after_thread_sync;");
        const int taddr = tmem_addr[0];

        if (tid < BLOCK_M) {
            const int tmem_row = cta_rank * 128 + warp_id * 32;
            half* row_ptr = C_ptr + (off_m + tid) * N + off_n;

            #pragma unroll 1
            for (int seg = 0; seg < 128; seg += 64) {
                const uint32_t row_base = (uint32_t)taddr + ((uint32_t)tmem_row << 16) + (uint32_t)(ACC_BASE + ACC1_OFF + seg);
                const uint32_t addr2_base = row_base + (uint32_t)ACC2_OFF;

                // Epilogue: pipelined 4x16 chunks per seg.
                float acc1a[16], acc2a[16];
                float acc1b[16], acc2b[16];

                // base 0 -> A
                tcgen05_ld_32x32bx8(acc1a + 0, (int)(row_base + 0));
                tcgen05_ld_32x32bx8(acc1a + 8, (int)(row_base + 8));
                tcgen05_ld_32x32bx8(acc2a + 0, (int)(addr2_base + 0));
                tcgen05_ld_32x32bx8(acc2a + 8, (int)(addr2_base + 8));

                // base 0
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                // prefetch base 16 -> B
                tcgen05_ld_32x32bx8(acc1b + 0, (int)(row_base + 16));
                tcgen05_ld_32x32bx8(acc1b + 8, (int)(row_base + 24));
                tcgen05_ld_32x32bx8(acc2b + 0, (int)(addr2_base + 16));
                tcgen05_ld_32x32bx8(acc2b + 8, (int)(addr2_base + 24));
                {
                    half2 h0 = silu_mul_h2(acc1a[0],  acc1a[1],  acc2a[0],  acc2a[1]);
                    half2 h1 = silu_mul_h2(acc1a[2],  acc1a[3],  acc2a[2],  acc2a[3]);
                    half2 h2 = silu_mul_h2(acc1a[4],  acc1a[5],  acc2a[4],  acc2a[5]);
                    half2 h3 = silu_mul_h2(acc1a[6],  acc1a[7],  acc2a[6],  acc2a[7]);
                    half2 h4 = silu_mul_h2(acc1a[8],  acc1a[9],  acc2a[8],  acc2a[9]);
                    half2 h5 = silu_mul_h2(acc1a[10], acc1a[11], acc2a[10], acc2a[11]);
                    half2 h6 = silu_mul_h2(acc1a[12], acc1a[13], acc2a[12], acc2a[13]);
                    half2 h7 = silu_mul_h2(acc1a[14], acc1a[15], acc2a[14], acc2a[15]);
                    const uint32_t u0 = bitcast_u32(h0);
                    const uint32_t u1 = bitcast_u32(h1);
                    const uint32_t u2 = bitcast_u32(h2);
                    const uint32_t u3 = bitcast_u32(h3);
                    const uint32_t u4 = bitcast_u32(h4);
                    const uint32_t u5 = bitcast_u32(h5);
                    const uint32_t u6 = bitcast_u32(h6);
                    const uint32_t u7 = bitcast_u32(h7);
                    const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                    const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                    const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                    const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                    stg_32b((const void*)(row_ptr + seg + 0), q0, q1, q2, q3);
                }

                // base 16
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                // prefetch base 32 -> A
                tcgen05_ld_32x32bx8(acc1a + 0, (int)(row_base + 32));
                tcgen05_ld_32x32bx8(acc1a + 8, (int)(row_base + 40));
                tcgen05_ld_32x32bx8(acc2a + 0, (int)(addr2_base + 32));
                tcgen05_ld_32x32bx8(acc2a + 8, (int)(addr2_base + 40));
                {
                    half2 h0 = silu_mul_h2(acc1b[0],  acc1b[1],  acc2b[0],  acc2b[1]);
                    half2 h1 = silu_mul_h2(acc1b[2],  acc1b[3],  acc2b[2],  acc2b[3]);
                    half2 h2 = silu_mul_h2(acc1b[4],  acc1b[5],  acc2b[4],  acc2b[5]);
                    half2 h3 = silu_mul_h2(acc1b[6],  acc1b[7],  acc2b[6],  acc2b[7]);
                    half2 h4 = silu_mul_h2(acc1b[8],  acc1b[9],  acc2b[8],  acc2b[9]);
                    half2 h5 = silu_mul_h2(acc1b[10], acc1b[11], acc2b[10], acc2b[11]);
                    half2 h6 = silu_mul_h2(acc1b[12], acc1b[13], acc2b[12], acc2b[13]);
                    half2 h7 = silu_mul_h2(acc1b[14], acc1b[15], acc2b[14], acc2b[15]);
                    const uint32_t u0 = bitcast_u32(h0);
                    const uint32_t u1 = bitcast_u32(h1);
                    const uint32_t u2 = bitcast_u32(h2);
                    const uint32_t u3 = bitcast_u32(h3);
                    const uint32_t u4 = bitcast_u32(h4);
                    const uint32_t u5 = bitcast_u32(h5);
                    const uint32_t u6 = bitcast_u32(h6);
                    const uint32_t u7 = bitcast_u32(h7);
                    const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                    const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                    const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                    const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                    stg_32b((const void*)(row_ptr + seg + 16), q0, q1, q2, q3);
                }

                // base 32
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                // prefetch base 48 -> B
                tcgen05_ld_32x32bx8(acc1b + 0, (int)(row_base + 48));
                tcgen05_ld_32x32bx8(acc1b + 8, (int)(row_base + 56));
                tcgen05_ld_32x32bx8(acc2b + 0, (int)(addr2_base + 48));
                tcgen05_ld_32x32bx8(acc2b + 8, (int)(addr2_base + 56));
                {
                    half2 h0 = silu_mul_h2(acc1a[0],  acc1a[1],  acc2a[0],  acc2a[1]);
                    half2 h1 = silu_mul_h2(acc1a[2],  acc1a[3],  acc2a[2],  acc2a[3]);
                    half2 h2 = silu_mul_h2(acc1a[4],  acc1a[5],  acc2a[4],  acc2a[5]);
                    half2 h3 = silu_mul_h2(acc1a[6],  acc1a[7],  acc2a[6],  acc2a[7]);
                    half2 h4 = silu_mul_h2(acc1a[8],  acc1a[9],  acc2a[8],  acc2a[9]);
                    half2 h5 = silu_mul_h2(acc1a[10], acc1a[11], acc2a[10], acc2a[11]);
                    half2 h6 = silu_mul_h2(acc1a[12], acc1a[13], acc2a[12], acc2a[13]);
                    half2 h7 = silu_mul_h2(acc1a[14], acc1a[15], acc2a[14], acc2a[15]);
                    const uint32_t u0 = bitcast_u32(h0);
                    const uint32_t u1 = bitcast_u32(h1);
                    const uint32_t u2 = bitcast_u32(h2);
                    const uint32_t u3 = bitcast_u32(h3);
                    const uint32_t u4 = bitcast_u32(h4);
                    const uint32_t u5 = bitcast_u32(h5);
                    const uint32_t u6 = bitcast_u32(h6);
                    const uint32_t u7 = bitcast_u32(h7);
                    const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                    const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                    const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                    const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                    stg_32b((const void*)(row_ptr + seg + 32), q0, q1, q2, q3);
                }

                // base 48
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                {
                    half2 h0 = silu_mul_h2(acc1b[0],  acc1b[1],  acc2b[0],  acc2b[1]);
                    half2 h1 = silu_mul_h2(acc1b[2],  acc1b[3],  acc2b[2],  acc2b[3]);
                    half2 h2 = silu_mul_h2(acc1b[4],  acc1b[5],  acc2b[4],  acc2b[5]);
                    half2 h3 = silu_mul_h2(acc1b[6],  acc1b[7],  acc2b[6],  acc2b[7]);
                    half2 h4 = silu_mul_h2(acc1b[8],  acc1b[9],  acc2b[8],  acc2b[9]);
                    half2 h5 = silu_mul_h2(acc1b[10], acc1b[11], acc2b[10], acc2b[11]);
                    half2 h6 = silu_mul_h2(acc1b[12], acc1b[13], acc2b[12], acc2b[13]);
                    half2 h7 = silu_mul_h2(acc1b[14], acc1b[15], acc2b[14], acc2b[15]);
                    const uint32_t u0 = bitcast_u32(h0);
                    const uint32_t u1 = bitcast_u32(h1);
                    const uint32_t u2 = bitcast_u32(h2);
                    const uint32_t u3 = bitcast_u32(h3);
                    const uint32_t u4 = bitcast_u32(h4);
                    const uint32_t u5 = bitcast_u32(h5);
                    const uint32_t u6 = bitcast_u32(h6);
                    const uint32_t u7 = bitcast_u32(h7);
                    const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                    const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                    const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                    const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                    stg_32b((const void*)(row_ptr + seg + 48), q0, q1, q2, q3);
                }
                
            }
        asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
        if (warp_id == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TOTAL_TMEM_COLS));
        }
    }

}

// ============================================================================
// Launch wrappers
// ============================================================================

template <int BLOCK_N, int BLOCK_M, int BLOCK_K, int NUM_STAGES>
at::Tensor dual_gemm_launch_collector(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    const int K = (int)A.size(1) * 2;
    const uint32_t dev = (uint32_t)A.get_device();

    auto A_ptr    = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr   = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr   = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr  = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr    = reinterpret_cast<half *>(C.data_ptr());

    const CUtensorMap& A_tmap  = get_ab_tmap_cached(A_ptr,  dev, (uint32_t)M, (uint32_t)K, (uint32_t)BLOCK_M,     (uint32_t)BLOCK_K);
    const CUtensorMap& B1_tmap = get_ab_tmap_cached(B1_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)(BLOCK_N/2), (uint32_t)BLOCK_K);
    const CUtensorMap& B2_tmap = get_ab_tmap_cached(B2_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)(BLOCK_N/2), (uint32_t)BLOCK_K);

    const CUtensorMap& SFA_tmap  = get_sf_tmap_cached(SFA_ptr,  dev, (uint32_t)M, (uint32_t)K, (uint32_t)BLOCK_K);
    const CUtensorMap& SFB1_tmap = get_sf_tmap_cached(SFB1_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)BLOCK_K);
    const CUtensorMap& SFB2_tmap = get_sf_tmap_cached(SFB2_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)BLOCK_K);

    constexpr int tb_size = BLOCK_M + 2 * WARP_SIZE;
    constexpr int A_size_c    = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size_c    = (BLOCK_N / 2) * BLOCK_K / 2;
    constexpr int SFA_size_c  = 128 * BLOCK_K / 16;
    constexpr int SFB_size_c  = 128 * BLOCK_K / 16;
    const int smem_size = (A_size_c + B_size_c + B_size_c + SFA_size_c + SFB_size_c + SFB_size_c) * NUM_STAGES;

    const int grid_m_clusters = M / (BLOCK_M * 2);
    const int grid_n_clusters = N / BLOCK_N;
    const int num_tiles = grid_m_clusters * grid_n_clusters;
    int clusters = num_tiles;
    if (clusters < 1) clusters = 1;
    dim3 grid(clusters * 2, 1, 1);

    auto kernel_fn = dual_gemm_cta2_collector_n64_kernel<BLOCK_M, BLOCK_K, NUM_STAGES>;
    static int max_smem = 0;
    if (smem_size > max_smem && smem_size > 48000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        max_smem = smem_size;
    }
    kernel_fn<<<grid, tb_size, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap, C_ptr, M, N, K);
    return C;
}

template <int BLOCK_N, int BLOCK_M, int BLOCK_K, int NUM_STAGES>
at::Tensor dual_gemm_launch_baseline(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    const int K = (int)A.size(1) * 2;
    const uint32_t dev = (uint32_t)A.get_device();

    auto A_ptr    = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr   = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr   = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr  = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr    = reinterpret_cast<half *>(C.data_ptr());

    const CUtensorMap& A_tmap  = get_ab_tmap_cached(A_ptr,  dev, (uint32_t)M, (uint32_t)K, (uint32_t)BLOCK_M,     (uint32_t)BLOCK_K);
    const CUtensorMap& B1_tmap = get_ab_tmap_cached(B1_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)(BLOCK_N/2), (uint32_t)BLOCK_K);
    const CUtensorMap& B2_tmap = get_ab_tmap_cached(B2_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)(BLOCK_N/2), (uint32_t)BLOCK_K);

    const CUtensorMap& SFA_tmap  = get_sf_tmap_cached(SFA_ptr,  dev, (uint32_t)M, (uint32_t)K, (uint32_t)BLOCK_K);
    const CUtensorMap& SFB1_tmap = get_sf_tmap_cached(SFB1_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)BLOCK_K);
    const CUtensorMap& SFB2_tmap = get_sf_tmap_cached(SFB2_ptr, dev, (uint32_t)N, (uint32_t)K, (uint32_t)BLOCK_K);

    constexpr int tb_size = BLOCK_M + 2 * WARP_SIZE;
    constexpr int A_size_c    = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size_c    = (BLOCK_N / 2) * BLOCK_K / 2;
    constexpr int SFA_size_c  = 128 * BLOCK_K / 16;
    constexpr int SFB_size_c  = 128 * BLOCK_K / 16;
    constexpr int PAD_c = 256;
    const int smem_size = (A_size_c + B_size_c + B_size_c + SFA_size_c + SFB_size_c + SFB_size_c + PAD_c) * NUM_STAGES;

    const int grid_m_clusters = M / (BLOCK_M * 2);
    const int grid_n_clusters = N / BLOCK_N;
    const int num_tiles = grid_m_clusters * grid_n_clusters;
    int clusters = num_tiles;
    if (clusters < 1) clusters = 1;
    dim3 grid(clusters * 2, 1, 1);

    auto kernel_fn = dual_gemm_cta2_baseline_n128_kernel<BLOCK_M, BLOCK_K, NUM_STAGES>;
    static int max_smem = 0;
    if (smem_size > max_smem && smem_size > 48000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        max_smem = smem_size;
    }
    kernel_fn<<<grid, tb_size, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap, C_ptr, M, N, K);
    return C;
}

at::Tensor dual_gemm(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int K = (int)A.size(1) * 2;
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    TORCH_CHECK((K % 256) == 0, "Unsupported K: ", K);
    TORCH_CHECK((M % 256) == 0, "Unsupported M: ", M);
    TORCH_CHECK((N % 64) == 0, "Unsupported N: ", N);

    if (M == 256) {
        if ((N == 3072 && K == 4096) || (N == 4096 && K == 7168)) {
            return dual_gemm_launch_collector<64, 128, 256, 7>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
        const int num_iters = K / 256;
        const int stages = (num_iters < 7) ? num_iters : 7;
        switch (stages) {
            case 1: return dual_gemm_launch_collector<64, 128, 256, 1>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 2: return dual_gemm_launch_collector<64, 128, 256, 2>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 3: return dual_gemm_launch_collector<64, 128, 256, 3>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 4: return dual_gemm_launch_collector<64, 128, 256, 4>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 5: return dual_gemm_launch_collector<64, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 6: return dual_gemm_launch_collector<64, 128, 256, 6>(A, B1, B2, SFA, SFB1, SFB2, C);
            default: return dual_gemm_launch_collector<64, 128, 256, 7>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
    } else {
        TORCH_CHECK((N % 128) == 0, "Unsupported N for N=128 path: ", N);
        if (M == 512 && K == 7168 && (N == 3072 || N == 4096)) {
            return dual_gemm_launch_baseline<128, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
        const int num_iters = K / 256;
        const int stages = (num_iters < 5) ? num_iters : 5;
        switch (stages) {
            case 1: return dual_gemm_launch_baseline<128, 128, 256, 1>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 2: return dual_gemm_launch_baseline<128, 128, 256, 2>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 3: return dual_gemm_launch_baseline<128, 128, 256, 3>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 4: return dual_gemm_launch_baseline<128, 128, 256, 4>(A, B1, B2, SFA, SFB1, SFB2, C);
            default: return dual_gemm_launch_baseline<128, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
    }
}

TORCH_LIBRARY(dual_gemm_13901_epi_pipe_module, m) {
    m.def("dual_gemm(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("dual_gemm", &dual_gemm);
}
"""


_compiled_module = None
_dual_gemm_op = None


def _get_module():
    global _compiled_module
    global _dual_gemm_op
    if _compiled_module is None:
        _compiled_module = load_inline(
            name="dual_gemm_13901_epi_pipe_cuda",
            cpp_sources="",
            cuda_sources=CUDA_SOURCE,
            functions=None,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--relocatable-device-code=false",
                "--extra-device-vectorization",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=False,
            is_python_module=False,
        )
        # Cache op handle to avoid repeated attribute resolution in the hot path.
        _dual_gemm_op = torch.ops.dual_gemm_13901_epi_pipe_module.dual_gemm
    return _compiled_module


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
    if _dual_gemm_op is None:
        _get_module()
    return _dual_gemm_op(a, b1, b2, sfa_permuted, sfb1_permuted, sfb2_permuted, c)

