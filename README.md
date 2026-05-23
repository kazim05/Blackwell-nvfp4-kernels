# Blackwell NVFP4 Kernel Hackathon

Submissions for the [Blackwell NVFP4 Kernel Hackathon](https://luma.com/9n27uem4), targeting the NVIDIA B200 (SM_100) architecture.

## Problems

| Directory | Task | Status |
|-----------|------|--------|
| [`nvfp4_dual_gemm/`](./nvfp4_dual_gemm/) | NVFP4 Dual GEMM with gated epilogue | Done |
| [`nvfp4_group_gemm/`](./nvfp4_group_gemm/) | NVFP4 Grouped GEMM (variable problem sizes, persistent kernel) | Done |

## Hardware Target

- **GPU**: NVIDIA B200 (Blackwell, SM_100)
- **Data type**: `float4_e2m1fn` (NVFP4) with `float8_e4m3fn` block scales
- **Accumulation**: FP16 output

## Realistic SOL Baselines

| Metric | Value |
|--------|-------|
| Memory BW | 5914.80 GB/s |
| nvfp4 compute | 5797.05 TFLOPS |

## Setup

```bash
pip install torch
# Kernels use PyTorch inline CUDA extensions — no separate build step needed.
```
