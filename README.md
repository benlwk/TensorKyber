<div align="left">

# TensorKyber

### Standalone NTT / iNTT Tensor-Core Benchmark for ML-KEM

This is the companion artifact for the paper "TensorKyber: Reformulating the NTT Implementation on Tensor Cores for High-Throughput Performance in ML-KEM". It contains a
self-contained CUDA benchmark of the **Number-Theoretic Transform (NTT) and its inverse (iNTT)** over the ML-KEM / Kyber ring

```
R_q = Z_q[x] / (x^256 + 1),   N = 256,   q = 3329
```

implemented by **reformulating the NTT layers as small INT8 matrix multiplications and
mapping them onto tensor cores** (WMMA `16x16x16`). The benchmark:

* runs four tensor-core NTT/iNTT variants,
* **validates every variant byte-exact** against an independent reference C NTT,
* reports **throughput (poly/s)** as mean, median and sample std-dev, plus median latency.

> [!NOTE]
> The ring `(n, q)` is identical for **all** ML-KEM parameter sets (ML-KEM-512/768/1024,
> a.k.a. Kyber-512/768/1024). The kernels are therefore parameter-independent; the `KYBER_K`
> switch only selects which *reference C namespace* is used for validation.

## Headline idea in one paragraph

A textbook Kyber NTT is 7 stages of butterflies. TensorKyber re-expresses each stage as a
**constant linear map** that fits a tensor-core tile: stages 1–4 collapse into one
`16x16` matrix `M1`, and stages 5–7 collapse into 16 independent per-block matrices
`M2^(m)` ("Stage-2"). Because INT8 tensor cores are integer-signed with limited range, every
constant matrix is stored as a **7-bit limb split** (lo/hi limbs) and the products are
recombined before a single modular reduction. The whole per-polynomial transform becomes a handful of `mma.sync` operations feeding a Barrett
reducer, while the inverse uses the mirror matrices `Minv` / `MA^(m)` and folds the final
`2^16` Montgomery scaling into `Minv`.

## Repository layout

```
ntt_bench/
├── ntt_bench.cu        # benchmark harness + the per-variant __global__ kernels
├── ntt_kernels.cuh     # NTT-only device kernels (reductions, matrices, cores)
├── ref/
│   ├── ntt.c           # reference C NTT / iNTT  (ground truth for validation)
│   ├── ntt.h
│   ├── params.h        # KYBER_N, KYBER_Q, reference-namespace selector
│   ├── reduce.c        # reference Montgomery / Barrett reductions
│   └── reduce.h
├── Makefile            # build + run
└── README.md           # this file
```

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA GPU with **INT8 tensor cores** (tested on sm_86, sm_89 and sm_90). |
| **CUDA** | CUDA Toolkit with `nvcc` (tested with CUDA 13.1). |
| **Host** | `gcc` (or a compatible C compiler) and GNU `make`. |

The current default target architecture is `sm_89`; override it for your GPU (see below).

## Build

```sh
make                 # builds ./ntt_bench 

## Run

```sh
make run                          # defaults: 262144 polynomials, 30 timed runs
make NPOLY=65536 RUNS=10 run      # smaller batch, fewer runs
make ARCH=sm_80 run               # build for a different GPU
```

Or drive the binary directly (positional arguments override the defaults):

```sh
./ntt_bench [npoly] [runs]        # defaults: 262144 30
```

### Overridable Makefile variables

| variable | default    | meaning |
|----------|-----------:|---------|
| `ARCH`   | `sm_89`    | `-arch` passed to `nvcc`. |
| `NPOLY`  | `262144`   | Number of polynomials in the batch. |
| `RUNS`   | `30`       | Timed runs per kernel (clamped to `MAX_RUNS = 64`). |




