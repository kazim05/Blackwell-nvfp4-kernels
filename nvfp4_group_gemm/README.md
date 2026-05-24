# NVFP4 Grouped GEMM - Worklog

**Task**: Execute G independent GEMM operations in a single kernel launch, each with a different problem size (M, N, K):

```
C[g] = A[g] @ B[g].T   for g in 0..G-1
```

All matrices are block-scaled NVFP4 (`float4_e2m1fn`) with `float8_e4m3fn` scale factors. Output is FP16.

---

## Submission History

- `submission_v1.py`: Baseline grouped GEMM (`kernel_group_gemm`). 2-stage TMA→MMA pipeline. One CTA per output tile — flat 1D grid over all tiles across all groups, with a `cta_to_group` lookup array mapping each CTA to its group and tile coordinates. 6 warps (192 threads): 4 epilogue + 1 TMA producer + 1 MMA consumer.

- `submission_v2.py`: Per-group kernel launch strategy (`kernel_single_gemm`). Same 2-stage kernel but called separately for each group to avoid divergence from variable problem sizes. Simpler CTA mapping at the cost of multiple launch overheads.

- `submission_v3.py`: Introduces the **persistent kernel** pattern (`persistent_group_gemm_kernel`). CTAs loop in a while loop over all tiles across all groups rather than terminating after one tile. Eliminates grid launch overhead; CTAs pick up new work as soon as they finish. 2-stage pipeline.

- `submission_v4.py`: Persistent kernel + **4-stage pipeline** (up from 2). Deeper staging hides TMA latency more aggressively. Barrier timeout doubled to match the longer pipeline depth.

- `submission_v5.py`: 4-stage persistent + **adaptive L2 cache eviction** — selects `EVICT_FIRST` vs `EVICT_LAST` per tile based on whether `M > N`, optimizing L2 reuse for the dominant dimension. Adds optimized binary search for group lookup. Grid capped at `min(total_tiles, max(num_sms * 3, num_sms))` for better SM saturation.

- `submission_v6.py`: All v5 features plus a major optimization pack:
  1. **Epilogue/mainloop overlap** via ping-pong TMEM — epilogue of tile N overlaps with MMA of tile N+1
  2. **SMEM-cached problem metadata** — group info loaded into shared memory once, avoiding repeated global reads per tile
  3. **Phase tracking across tiles** — eliminates barrier reinit overhead between tiles in persistent loop
  4. **Interleaved `tcgen05.cp` with MMA** — scale factor copy overlapped with matrix multiply
  5. **Vectorized epilogue stores** — wider FP16 stores to global memory
  6. Grid tightened to `min(total_tiles, num_sms)` for exact SM occupancy

- `submission_v7.py`: All v6 features + **NUM_STAGES tuned from 4→3** for a better SMEM vs latency-hiding balance. Refactors host-side parameters into a `LaunchParams` struct (`h_A_tmaps`, `h_B_tmaps`, `h_problems` pre-bundled) and copies it to device with a single `cudaMemcpyAsync`. **Best leaderboard result: runtime 25.407 µs (775 TFLOPS).**

---

## Key Design Decisions

- **Warp specialization**: 6 warps (192 threads) — warp 4 = TMA producer, warp 5 = MMA consumer, warps 0–3 = epilogue
- **Persistent kernel**: CTAs loop over all tiles across all groups; SM utilization stays high regardless of irregular group sizes
- **Adaptive cache eviction**: `EVICT_FIRST` for the smaller dimension, `EVICT_LAST` for the larger — reduces L2 pollution from non-reused data
- **SMEM-cached group metadata**: `ProblemInfo` array loaded into shared memory to avoid per-tile global reads
- **Ping-pong TMEM**: epilogue reads from one TMEM buffer while MMA writes to the other, hiding epilogue latency
- **3-stage pipeline**: 4 stages (~144KB SMEM) left only ~84KB for L1 cache; 3 stages (~108KB) reduces L1 pressure, which outweighs the benefit of one extra prefetch stage for grouped workloads

---

## Leaderboard Results

Unit: microseconds (lower is better). Best submission: `submission_v7.py` with runtime **25.407 µs** (**775 TFLOPS**).

| Version | Runtime (µs) | TFLOPS  | Notes |
|---------|-------------|---------|-------|
| SOL     | 5.211       | 3,777   | Speed-of-light (B200 peak) |
| v1      | —           | —       | Z-dim grid baseline |
| v2      | —           | —       | Per-group launch |
| v3      | —           | —       | Persistent kernel |
| v4      | —           | —       | + 4-stage pipeline |
| v5      | —           | —       | + adaptive cache eviction |
| v6      | —           | —       | + TMEM overlap + SMEM cache |
| v7      | **25.407**  | **775** | + 3-stage tuning + LaunchParams — best |

> TFLOPS = geomean(2·Σ Mₘ·Nₘ·Kₘ) / runtime; sums FLOPS across all G groups per benchmark. SOL runtime computed from per-shape times in task.yml (18.833, 10.667, 2.406, 1.525 µs).

---

## Files

| File | Description |
|------|-------------|
| `task.py` | Task type definitions (`input_t`, `output_t`, `TestSpec`) |
| `task.yml` | Task metadata — problem sizes, benchmarks, ranking |
| `reference.py` | PyTorch reference using `torch._scaled_mm` |
| `utils.py` | Shared utilities (allclose checks, seeding, L2 cache flush) |
| `submission_v1.py` | Baseline Z-dim grouped GEMM, 2-stage pipeline |
| `submission_v2.py` | Per-group kernel launch, 2-stage pipeline |
| `submission_v3.py` | Persistent kernel, 2-stage pipeline |
| `submission_v4.py` | Persistent + 4-stage pipeline |
| `submission_v5.py` | + adaptive cache eviction + optimized group lookup |
| `submission_v6.py` | + ping-pong TMEM + SMEM-cached problems + vectorized epilogue |
| `submission_v7.py` | + 3-stage tuning + LaunchParams struct — best result |

---

## Notes

- Persistent kernel avoids re-launching for each group but requires careful phase/barrier management across tile transitions
- 4 stages used ~144KB of SMEM leaving only ~84KB for L1 cache; tuning to 3 stages (~108KB) reduced L1 pressure, which matters more for grouped GEMM than an extra prefetch stage
- Binary search for group lookup is critical for large group counts — linear scan dominates for G > 8
- `LaunchParams` struct reduces kernel argument size and avoids repeated host-side setup per call
