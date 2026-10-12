# Standalone NTT / iNTT benchmark -- Kyber-512

Self-contained extraction of the NTT/iNTT benchmark from
`../bench_ntt.cu`, trimmed to the three production tensor-core batch paths.

## Layout

| file                | role                                                            |
|---------------------|-----------------------------------------------------------------|
| `ntt_bench.cu`      | benchmark (ported from `../bench_ntt.cu`)                       |
| `ntt_kernels.cuh`   | NTT-only device kernels: reductions, TC matrices, the compute cores `ntt_core`/`intt_core`, the register-staged wrappers `ntt_regstage`/`intt_regstage`, `stage_tc_matrices`/`setup_tc_matrices`. `#include`d by the benchmark. All KEM code (Keccak/SHAKE, CBD, matrix generation, poly/polyvec packing, CPA/KEM kernels and the KEM harness) has been removed. |
| `ref/ntt.c`         | reference C NTT (`ref/` ground truth), compiled with `-DKYBER_K=$(K)` |
| `ref/reduce.c`      | reference Montgomery/Barrett reductions needed by `ref/ntt.c`   |
| `Makefile`          | build + run                                                     |

## Variants benchmarked

1. **2TC** - register-staged, WMMA per-block Stage-2. Each lane holds its own
   polynomial and the 16-slot shared buffer is staged out of registers before
   running the shared compute core.
2. **TC-CC** - register-staged, scalar-butterfly Stage-2. Same staging as (1).
3. **TC-CC-CP** - cp.async single-buffer staging (16-slot buffer per warp),
   scalar-butterfly Stage-2.
4. **TC-PACK** - packed forward NTT on 2-bit packed ternary input
   (4 coeffs/byte, 64 B/poly) cp.async'd into shared, unpacked into the standard
   int16 slot layout, forward core with the BASIC (single-limb) `M1` matrix and
   scalar Stage-2. Its inverse is TC-CC-CP.

Removed relative to the original benchmark: Baseline (BFS, `ref` source),
EDFS-1-6, the cp.async WMMA-S2 variant and all single-warp
(1 thread / 1 warp per polynomial) variants.

All four run in the "32 tasks per warp" geometry with `block = 64`; every
polynomial is one task.

## Build & run

```sh
make                 # build ./ntt_bench  (Kyber-512 by default)
make run             # 262144 polynomials, 30 timed runs
```

Overrides:

```sh
make NPOLY=65536 RUNS=10 run    # smaller batch / fewer runs
make K=3 run                    # Kyber-768 instead of Kyber-512
make ARCH=sm_89                 # target architecture (default sm_89)
```

Positional arguments are supported too: `./ntt_bench [npoly] [runs]`.
