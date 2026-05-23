import torch
from task import input_t, output_t
import os
import subprocess
import tempfile
import shutil

CUDA_SRC = r"""
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

// Cache eviction policies from CUTLASS
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;

// Descriptor encoding helper
__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

// Warp uniform helper
__device__ inline
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

// Barrier operations
__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@!P1 bra.uni LAB_WAIT;\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

// TMA operations
__device__ inline
void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

// TCGEN05 operations for NVFP4
__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline
void tcgen05_mma_nvfp4(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  const int d_tmem = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// TMEM load operations
struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";
  static constexpr char _16x128b[] = ".16x128b";
  static constexpr char _16x256b[] = ".16x256b";
};

template <int NUM_REGS, const char *SHAPE, int NUM>
__device__ inline
void tcgen05_ld(float *tmp, int row, int col) {
  int addr = (row << 16) | col;

  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
}

__device__ inline void tcgen05_ld_16x256bx4(float *tmp, int row, int col) {
  tcgen05_ld<16, SHAPE::_16x256b, 4>(tmp, row, col);
}

void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
    error_msg_ptr = "unable to get error string";
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

void init_AB_tmap(
  CUtensorMap *tmap,
  const char *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width
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

// Kernel configuration
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;
constexpr int NUM_STAGES = 2;
constexpr int NUM_WARPS = 6;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__global__
__launch_bounds__(TB_SIZE)
void kernel_group_gemm(
  const CUtensorMap * const __restrict__ __grid_constant__ A_tmaps,
  const CUtensorMap * const __restrict__ __grid_constant__ B_tmaps,
  const char ** const __restrict__ __grid_constant__ SFA_ptrs,
  const char ** const __restrict__ __grid_constant__ SFB_ptrs,
  half ** const __restrict__ __grid_constant__ C_ptrs,
  const int * const __restrict__ __grid_constant__ problem_sizes,
  const int * const __restrict__ __grid_constant__ cta_to_group,
  const int * const __restrict__ __grid_constant__ cta_offsets,
  const int __grid_constant__ total_tiles,
  const int __grid_constant__ num_groups,
  const int __grid_constant__ K
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  int group_idx = cta_to_group[bid];
  int group_offset = cta_offsets[group_idx];
  int cta_in_group = bid - group_offset;

  const int M = problem_sizes[group_idx * 4 + 0];
  const int N = problem_sizes[group_idx * 4 + 1];
  const int group_K = problem_sizes[group_idx * 4 + 2];

  const int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
  const int bid_m = cta_in_group / grid_n;
  const int bid_n = cta_in_group % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                :: "r"(smem), "r"(BLOCK_N * 2));
  }
  __syncthreads();

  const int num_iters = group_K / BLOCK_K;

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const int off_k = iter_k * BLOCK_K;
      tma_3d_gmem2smem(A_smem, &A_tmaps[group_idx], 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, &B_tmaps[group_idx], 0, off_n, off_k / 256, mbar_addr, cache_B);

      const int rest_k = group_K / 16 / 4;
      const char *SFA_src = SFA_ptrs[group_idx] + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;
      const char *SFB_src = SFB_ptrs[group_idx] + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; iter_k++)
      issue_tma(iter_k, iter_k);

    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }
  }
  else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U)
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U);

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      const uint64_t SFA_desc = make_desc_SF(SFA_smem);
      const uint64_t SFB_desc = make_desc_SF(SFB_smem);

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }

      for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }

      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    }

    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  else if (tid < BLOCK_M) {
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    half *C_ptr = C_ptrs[group_idx];

    for (int m = 0; m < 32 / 16; m++)
      for (int n = 0; n < BLOCK_N / 8; n++) {
        float tmp[4];
        tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, n * 8);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
        const int col = off_n + n * 8 + (lane_id % 4) * 2;

        if (row < M && col < N) {
          half2 out = __float22half2_rn({tmp[0], tmp[1]});
          reinterpret_cast<half2 *>(C_ptr + row * N + col)[0] = out;
        }
        if (row + 8 < M && col < N) {
          half2 out = __float22half2_rn({tmp[2], tmp[3]});
          reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = out;
        }
      }

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                  :: "r"(0), "r"(BLOCK_N * 2));
  }
}

at::Tensor kernel_launch(
  const std::vector<at::Tensor>& A_tensors,
  const std::vector<at::Tensor>& B_tensors,
  const std::vector<at::Tensor>& SFA_tensors,
  const std::vector<at::Tensor>& SFB_tensors,
  const std::vector<at::Tensor>& C_tensors,
  const std::vector<std::tuple<int, int, int, int>>& problem_sizes
) {
  int num_groups = A_tensors.size();
  int K = std::get<2>(problem_sizes[0]);

  std::vector<CUtensorMap> A_tmaps(num_groups);
  std::vector<CUtensorMap> B_tmaps(num_groups);
  std::vector<const char*> SFA_ptrs(num_groups);
  std::vector<const char*> SFB_ptrs(num_groups);
  std::vector<half*> C_ptrs(num_groups);

  std::vector<int> problem_sizes_flat(num_groups * 4);
  std::vector<int> cta_offsets(num_groups);
  int total_tiles = 0;

  for (int g = 0; g < num_groups; g++) {
    int M = std::get<0>(problem_sizes[g]);
    int N = std::get<1>(problem_sizes[g]);
    int group_K = std::get<2>(problem_sizes[g]);

    init_AB_tmap(&A_tmaps[g], reinterpret_cast<const char*>(A_tensors[g].data_ptr()), M, group_K, BLOCK_M, BLOCK_K);
    init_AB_tmap(&B_tmaps[g], reinterpret_cast<const char*>(B_tensors[g].data_ptr()), N, group_K, BLOCK_N, BLOCK_K);

    SFA_ptrs[g] = reinterpret_cast<const char*>(SFA_tensors[g].data_ptr());
    SFB_ptrs[g] = reinterpret_cast<const char*>(SFB_tensors[g].data_ptr());
    C_ptrs[g] = reinterpret_cast<half*>(C_tensors[g].data_ptr());

    problem_sizes_flat[g * 4 + 0] = M;
    problem_sizes_flat[g * 4 + 1] = N;
    problem_sizes_flat[g * 4 + 2] = group_K;
    problem_sizes_flat[g * 4 + 3] = 1;

    cta_offsets[g] = total_tiles;
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    total_tiles += grid_m * grid_n;
  }

  std::vector<int> cta_to_group(total_tiles);
  for (int g = 0; g < num_groups; g++) {
    int M = std::get<0>(problem_sizes[g]);
    int N = std::get<1>(problem_sizes[g]);
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    for (int i = 0; i < grid_m * grid_n; i++) {
      cta_to_group[cta_offsets[g] + i] = g;
    }
  }

  CUtensorMap *d_A_tmaps, *d_B_tmaps;
  const char **d_SFA_ptrs, **d_SFB_ptrs;
  half **d_C_ptrs;
  int *d_problem_sizes, *d_cta_to_group, *d_cta_offsets;

  check_cuda(cudaMalloc(&d_A_tmaps, num_groups * sizeof(CUtensorMap)));
  check_cuda(cudaMalloc(&d_B_tmaps, num_groups * sizeof(CUtensorMap)));
  check_cuda(cudaMalloc(&d_SFA_ptrs, num_groups * sizeof(const char*)));
  check_cuda(cudaMalloc(&d_SFB_ptrs, num_groups * sizeof(const char*)));
  check_cuda(cudaMalloc(&d_C_ptrs, num_groups * sizeof(half*)));
  check_cuda(cudaMalloc(&d_problem_sizes, num_groups * 4 * sizeof(int)));
  check_cuda(cudaMalloc(&d_cta_to_group, total_tiles * sizeof(int)));
  check_cuda(cudaMalloc(&d_cta_offsets, num_groups * sizeof(int)));

  check_cuda(cudaMemcpy(d_A_tmaps, A_tmaps.data(), num_groups * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_B_tmaps, B_tmaps.data(), num_groups * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_SFA_ptrs, SFA_ptrs.data(), num_groups * sizeof(const char*), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_SFB_ptrs, SFB_ptrs.data(), num_groups * sizeof(const char*), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_C_ptrs, C_ptrs.data(), num_groups * sizeof(half*), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_problem_sizes, problem_sizes_flat.data(), num_groups * 4 * sizeof(int), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_cta_to_group, cta_to_group.data(), total_tiles * sizeof(int), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_cta_offsets, cta_offsets.data(), num_groups * sizeof(int), cudaMemcpyHostToDevice));

  constexpr int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  constexpr int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  constexpr int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = kernel_group_gemm;
  if (smem_size > 48000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  this_kernel<<<total_tiles, TB_SIZE, smem_size>>>(
    d_A_tmaps,
    d_B_tmaps,
    d_SFA_ptrs,
    d_SFB_ptrs,
    d_C_ptrs,
    d_problem_sizes,
    d_cta_to_group,
    d_cta_offsets,
    total_tiles,
    num_groups,
    K
  );

  cudaFree(d_A_tmaps);
  cudaFree(d_B_tmaps);
  cudaFree(d_SFA_ptrs);
  cudaFree(d_SFB_ptrs);
  cudaFree(d_C_ptrs);
  cudaFree(d_problem_sizes);
  cudaFree(d_cta_to_group);
  cudaFree(d_cta_offsets);

  return C_tensors[0];
}

PYBIND11_MODULE(group_gemm_module, m) {
    m.def("kernel_launch", &kernel_launch, "Group GEMM kernel launch");
}
"""

import os
import shutil

# Manual compilation to ensure sm_100a is used (PyTorch's load_inline adds conflicting sm_100 flags)
def compile_cuda_module():
    import sys
    import hashlib
    
    # Create a unique build directory based on source hash
    src_hash = hashlib.md5(CUDA_SRC.encode()).hexdigest()[:8]
    build_dir = f'/tmp/group_gemm_build_{src_hash}'
    os.makedirs(build_dir, exist_ok=True)
    
    so_name = 'group_gemm_module.cpython-313-x86_64-linux-gnu.so'
    so_path = os.path.join(build_dir, so_name)
    
    # Check if already compiled
    if os.path.exists(so_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("group_gemm_module", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    
    # Write CUDA source
    cuda_path = os.path.join(build_dir, 'kernel.cu')
    with open(cuda_path, 'w') as f:
        f.write(CUDA_SRC)
    
    # Get paths
    torch_lib = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_lib, 'include')
    torch_api_include = os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')
    python_include = subprocess.check_output([sys.executable, '-c', 
        'import sysconfig; print(sysconfig.get_path("include"))']).decode().strip()
    cuda_include = '/usr/local/cuda/include'
    
    # Compile CUDA with sm_100a
    obj_path = os.path.join(build_dir, 'kernel.o')
    nvcc_cmd = [
        '/usr/local/cuda/bin/nvcc',
        '-O3', '--use_fast_math', '-lineinfo',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a',  # Explicit gencode for sm_100a
        '-Xcompiler', '-fPIC',
        '-DTORCH_EXTENSION_NAME=group_gemm_module',
        '-DTORCH_API_INCLUDE_EXTENSION_H',
        '-isystem', torch_include,
        '-isystem', torch_api_include,
        '-isystem', cuda_include,
        '-isystem', python_include,
        '-c', cuda_path,
        '-o', obj_path
    ]
    print(f"Running: {' '.join(nvcc_cmd)}")
    subprocess.run(nvcc_cmd, check=True)
    
    # Link into shared library with pybind11
    link_cmd = [
        'c++',
        '-shared',
        '-fPIC',
        '-o', so_path,
        obj_path,
        f'-L{os.path.join(torch_lib, "lib")}',
        '-ltorch', '-ltorch_cpu', '-ltorch_python', '-lc10', '-lc10_cuda', '-ltorch_cuda',
        '-L/usr/local/cuda/lib64',
        '-lcudart', '-lcuda',
        f'-Wl,-rpath,{os.path.join(torch_lib, "lib")}',
        '-Wl,-rpath,/usr/local/cuda/lib64'
    ]
    print(f"Running: {' '.join(link_cmd)}")
    subprocess.run(link_cmd, check=True)
    
    # Load the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("group_gemm_module", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

module = compile_cuda_module()

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    A_tensors = [a for a, b, c in abc_tensors]
    B_tensors = [b for a, b, c in abc_tensors]
    C_tensors = [c for a, b, c in abc_tensors]

    SFA_tensors = [sfa for sfa, sfb in sfasfb_reordered_tensors]
    SFB_tensors = [sfb for sfa, sfb in sfasfb_reordered_tensors]

    module.kernel_launch(A_tensors, B_tensors, SFA_tensors, SFB_tensors, C_tensors, problem_sizes)

    return C_tensors
