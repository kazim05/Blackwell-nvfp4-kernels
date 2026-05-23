# NVFP4 Dual GEMM - Worklog

**Task**: Given a single A matrix and two B matrices (B1, B2), compute:
- `C1 = A @ B1.T` (with block-scaled FP4)
- `C2 = A @ B2.T` (with block-scaled FP4)
- Apply gated epilogue: `out = silu(C1) * C2`

All in a single fused kernel to avoid reloading A from global memory twice.

---

## Submission History

- `submission_v1.py`: Hand-written PTX/inline CUDA via `torch.utils.cpp_extension.load_inline`. Baseline kernel — loads A and both B tiles, computes `A @ B1` and `A @ B2` in one threadblock, applies gated epilogue.
- `submission_v2.py`: CuteDSL kernel (initial). 128 threads per CTA. Uses CUTLASS CuteDSL pipeline abstractions instead of raw PTX.
- `submission_v3.py`: CuteDSL + warp specialization (ported pattern from `nvfp4_gemm`). Separates TMA producer and MMA consumer warps for better pipelining.
- `submission_v4.py`: CuteDSL + warp specialization + enhanced TMA multicasting with `(2,2)` cluster shape for 4x reduction in memory traffic.

**Best result**: `submission_v4.py` — geomean **20.692 µs** across the four benchmark shapes.

---

## Key Design Decisions

- **Warp specialization**: TMA producer warp + MMA consumer warp + epilogue warps (6 warps total, 192 threads)
- **Activation stationary**: A tile stays in SMEM across both B1 and B2 MMAs to halve A bandwidth
- **Scale factors**: Reordered to match TCGen05 consumption pattern; 1D TMA loads to SMEM then `tcgen05.cp` to TMEM
- **Epilogue**: `silu` gating fused into epilogue; uses `16x256b` or `32x32b` tcgen05 load depending on `BLOCK_N`

---

## Leaderboard Results

Unit: microseconds. Best submission: `submission_v4.py` with geomean **20.692 µs**.

| Version | Geomean (µs) | Notes |
|---------|-------------|-------|
| SOL     | 4.886       | Speed-of-light (B200 peak) |
| v1      | —           | PTX baseline |
| v2      | —           | CuteDSL initial |
| v3      | —           | + warp specialization |
| v4      | **20.692**  | + (2,2) TMA multicast cluster |

---

## Files

| File | Description |
|------|-------------|
| `task.py` | Task harness — correctness check and timing loop |
| `task.yml` | Task metadata (problem sizes, benchmarks) |
| `reference.py` | Naive PyTorch reference implementation |
| `submission_v1.py` | Hand-written PTX kernel via `load_inline` |
| `submission_v2.py` | CuteDSL kernel (initial, 128 threads/CTA) |
| `submission_v3.py` | CuteDSL + warp specialization |
| `submission_v4.py` | CuteDSL + warp spec + `(2,2)` TMA multicast cluster |
| `utils.py` | Shared utilities (tensormap init, barrier helpers, scale reordering) |

---

## Notes

- Using tensormap for SF is slower than direct pointer arithmetic in some cases
- `tcgen05.wait::ld` must be issued after all `tcgen05.ld` in the epilogue warp
- Avoid `R2UR` instructions — make `bid` and `warp_id` warp-uniform via `__shfl_sync()`
