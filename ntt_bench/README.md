# Standalone NTT / iNTT benchmark -- Kyber-512

Self-contained extraction of the NTT/iNTT benchmark from
`../bench_ntt.cu`, trimmed to the three production tensor-core batch paths.

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

## Build & run

```sh
make                 # build ./ntt_bench  (Kyber-512 by default)
make run             # 262144 polynomials, 30 timed runs
```

Overrides:

```sh
make NPOLY=65536 RUNS=10 run    # smaller batch / fewer runs
make ARCH=sm_89                 # target architecture (default sm_89)
```

Positional arguments are supported too: `./ntt_bench [npoly] [runs]`.
