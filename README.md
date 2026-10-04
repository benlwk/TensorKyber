<div align="center">

# TensorKyber

### Standalone NTT / iNTT Tensor-Core Benchmark for ML-KEM

*Companion artifact for the paper*

**"TensorKyber: Reformulating the NTT Implementation on Tensor Cores for High-Throughput Performance in ML-KEM"**

`CUDA` · `WMMA / INT8 tensor cores` · `N = 256, q = 3329` · `byte-exact vs. reference C`

</div>

---

## About

This directory is the **reproduction artifact** for the paper *TensorKyber*. It contains a
self-contained CUDA benchmark of the **Number-Theoretic Transform (NTT) and its inverse
(iNTT)** over the ML-KEM / Kyber ring

```
R_q = Z_q[x] / (x^256 + 1),   N = 256,   q = 3329
```

implemented by **reformulating the NTT layers as small INT8 matrix multiplications and
mapping them onto tensor cores** (WMMA `16x16x16`). The benchmark:

* runs four tensor-core NTT/iNTT variants,
* **validates every variant byte-exact** against an independent reference C NTT,
* reports **throughput (poly/s)** as mean, median and sample std-dev, plus median latency.

It is intentionally minimal: all KEM/CPA machinery (Keccak/SHAKE, CBD, matrix
generation, key/ciphertext packing) has been stripped out, so the code reads as *exactly*
the NTT reformulation discussed in the paper.

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
recombined with weights `1, 128, 16384` before a single modular reduction. The whole
per-polynomial transform becomes a handful of `mma.sync` operations feeding a Barrett
reducer, while the inverse uses the mirror matrices `Minv` / `MA^(m)` and folds the final
`2^16` Montgomery scaling into `Minv`.

## Results at a glance

Full results, plots and analysis are in the paper. The table below is a *sample* run on a
single laptop GPU (`NVIDIA RTX 4050`, `sm_89`, CUDA 13.1) with `4096` polynomials and `3`
timed runs — it is meant to show the output shape, **not** to be a benchmark result:

| Variant   | NTT (M poly/s) | iNTT (M poly/s) | Round-trip (M poly/s) | Correctness |
|-----------|---------------:|----------------:|----------------------:|:-----------:|
| 2TC       |     76.8       |     83.3        |     39.9              | byte-exact  |
| TC-CC     |     83.0       |     72.6        |     38.7              | byte-exact  |
| TC-CC-CP  |     86.9       |     84.5        |     42.8              | byte-exact  |
| TC-PACK   |     94.8       |     84.7        |     44.7              | byte-exact  |

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

| file                | role |
|---------------------|------|
| `ntt_bench.cu`      | Host harness (validation + timing + summary) and the `__global__` entry kernels `k_ntt_2tc_tc_cc`, `k_invntt_2tc_tc_cc`, `k_ntt_tc_cc_cp`, `k_invntt_tc_cc_cp`, `k_ntt_tc_pack`. |
| `ntt_kernels.cuh`   | The reusable device layer: mod-q reductions, twiddles `d_zetas`, the constant matrices, the compute cores `ntt_core` / `intt_core`, the register-staged wrappers `ntt_regstage` / `intt_regstage`, and `stage_tc_matrices` / `setup_tc_matrices`. |
| `ref/*`             | The independent reference C NTT/iNTT used as ground truth; compiled with the same `-DKYBER_K`. |
| `Makefile`          | Builds the reference objects with `gcc`, then links the benchmark with `nvcc`. |

## How the transform is mapped to tensor cores

Everything below lives in `ntt_kernels.cuh`.

**1. Constant matrices, 7-bit limb split.** Each NTT stage is a fixed linear map over
`Z_q`. TensorKyber stores each map as two INT8 matrices, `lo` and `hi`, that satisfy
`M = lo + 128 * hi`. For a forward transform these are `M1` (stages 1–4, one `16x16` tile)
and the 16 per-block `M2^(m)` (stages 5–7). The inverse uses `Minv` and `MA^(m)`. `M1`/`Minv`
are staged from `__constant__` into shared memory (`stage_tc_matrices`); the Stage-2 matrices
live in `__device__` globals because `wmma::load_matrix_sync` cannot read `__constant__`.

**2. Provenance of the matrices.** `setup_tc_matrices()` builds every matrix on the host by
**probing the reference layer code with unit vectors**, so each matrix provably equals the
real linear map — no hand-derived constants to get wrong.

**3. The core.** `ntt_core<BASIC, S2>()` runs on a buffer of 16 staged polynomials (one per
`16`-coefficient block, `TC_SLOT_STRIDE = N + 8` apart). It applies `M1` to each block with
`wmma::mma_sync`, recombines the accumulator with weights `1 / 128 / 16384`, and reduces mod
`q` once with Barrett. Stage 2 is then either per-block WMMA tiles (`S2 = 1`) or DFS-ordered
scalar butterflies (`S2 = 0`). `intt_core<S2>()` is the exact mirror, with the final `2^16`
Montgomery scaling folded into `Minv`.

**4. Reductions.** Montgomery (`mont_reduce`) and Barrett (`barrett`, `barrett32`) cover the
scalar path; `barrett32u` (unsigned Barrett via `__umulhi`) normalises the wide tensor-core
accumulator `S = Σ A·B` (16 terms of `(q-1)^2`) exactly back into `[0, q)`.

**5. Data movement.** Register-staged variants keep each lane's polynomial in registers and
stage 16 polynomials per round into shared memory. The `-CP` / `TC-PACK` variants instead use
**`cp.async`** to stream a 16-slot shared buffer straight from global memory, hiding latency
with a single-buffered pipeline.

## Benchmark variants

All four run in the **"32 tasks per warp"** geometry with `block = 64`; each polynomial is one
task.

| Variant       | Input        | Staging                  | Stage-2 (layers 5–7)   | Notes |
|---------------|--------------|--------------------------|------------------------|-------|
| **2TC**       | `int16`      | register → shared        | **WMMA** per-block tile | Fewest instructions in Stage-2. |
| **TC-CC**     | `int16`      | register → shared        | scalar DFS butterflies | Compare against 2TC. |
| **TC-CC-CP**  | `int16`      | **`cp.async`** single-buf | scalar DFS butterflies | Also the **inverse** used by TC-PACK. |
| **TC-PACK**   | **packed 2-bit ternary** (4 coeffs/byte, 64 B/poly) | `cp.async` + unpack to `int16` | scalar DFS butterflies | Forward-only; smaller input, single-limb `BASIC` `M1` (2 `mma`/tile instead of 4). |

`TC-PACK` exists to show the win from feeding the transform its **native small-coefficient
form**: the packed input also lets the first `M1` use the single-limb (`BASIC`) path. The
benchmark times `TC-CC-CP`'s forward on the *same* small input so the packing benefit is a
fair head-to-head.

> Removed relative to the original research harness: the BFS baseline, EDFS-1–6, the
> `cp.async` WMMA-Stage-2 variant, and all single-warp (one thread / one warp per polynomial)
> variants.

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA GPU with **INT8 tensor cores** (Volta `sm_70`+ for WMMA; tested on `sm_89`). |
| **CUDA** | CUDA Toolkit with `nvcc` and the `mma.h` / `cuda_pipeline_primitives.h` headers (tested with CUDA 13.1). |
| **Host** | `gcc` (or a compatible C compiler) and GNU `make`. |

The current default target architecture is `sm_89`; override it for your GPU (see below).

## Build

```sh
make                 # builds ./ntt_bench  (Kyber-512 by default)
```

The build is two-stage: the reference C objects (`ntt.o`, `reduce.o`) are compiled with `gcc`,
then `ntt_bench.cu` is linked against them with `nvcc`.

## Run

```sh
make run                          # defaults: 262144 polynomials, 30 timed runs
make K=3 run                      # Kyber-768 instead of Kyber-512
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
| `K`      | `2`        | ML-KEM parameter set: `2` = ML-KEM-512, `3` = ML-KEM-768, `4` = ML-KEM-1024. |
| `NPOLY`  | `262144`   | Number of polynomials in the batch. |
| `RUNS`   | `30`       | Timed runs per kernel (clamped to `MAX_RUNS = 64`). |

### Example output

```
==============================================================================
 Standalone NTT / iNTT benchmark -- Kyber-512 (N=256, q=3329)
 GPU: NVIDIA GeForce RTX 4050 Laptop GPU (sm_89, 20 SMs)
 batch = 262144 polynomials, 30 timed runs (mean & median reported)
 variants: 2TC | TC-CC | TC-CC-CP | TC-PACK
==============================================================================

--- Correctness vs reference C (first 64 polys, mod-q normalised) ---
  2TC                  NTT ok (0 bad)   iNTT ok (0 bad)

--- Batch path (32 tasks/warp, 2TC NTT / 2TC iNTT, block=64) ---
  NTT  batch           mean     ... poly/s   median     ... poly/s   +/- ...   lat   0.0750 ms
  iNTT batch           mean     ... poly/s   median     ... poly/s   +/- ...   lat   0.0640 ms
  Round-trip (batch)      mean     ... poly/s   median     ... poly/s
...
  all variants byte-exact vs the reference C NTT/iNTT

==========================================================================================
 SUMMARY: NTT/iNTT batch variants -- mean & median over 30 runs, 262144 polys (M poly/s)
==========================================================================================
Variant     geometry        block |  NTT mean   NTT med | iNTT mean  iNTT med |   RT mean    RT med |  NTT  iNTT
...
```

## Benchmark protocol

* **Correctness first.** Before any timing, each variant's forward and inverse output is
  compared against the reference C NTT/iNTT (mod `q` normalised) on the first `min(npoly, 64)`
  polynomials. A non-zero mismatch count is reported and flags the timings as untrustworthy.
* **Warm-up.** Every kernel is launched twice, untimed, before measurement.
* **Structurally identical loops.** Every variant **re-uploads its input** before each timed
  forward run, so each run starts from the same input/cache/power state and the four loops are
  directly comparable. (Without re-uploading the packed buffer, TC-PACK's forward throughput
  swings enough to make its mean and median diverge by ~20%.)
* **Reported statistics.** Per variant: mean, median and sample std-dev of polynomial
  throughput (poly/s), plus median latency in ms; the summary table prints mean *and* median
  for NTT, iNTT, and the NTT+iNTT round trip.
* **Exit code** is non-zero if any variant deviates from the reference.

## Reproducing the paper

1. Build for your target GPU: `make ARCH=sm_XX`.
2. Run with the paper's batch size and run count (see the *Evaluation* section of the paper),
   e.g. `make NPOLY=262144 RUNS=30 run`.
3. Confirm the line `all variants byte-exact vs the reference C NTT/iNTT` appears.
4. Read the **SUMMARY** table for the per-variant NTT / iNTT / round-trip throughput.

Because the kernels are ML-KEM-parameter-independent, running with `K=2`, `K=3` or `K=4`
changes only which reference namespace performs validation — the measured kernels are the same.

## Citation

If you use this artifact, please cite the paper:

```bibtex
@article{tensorkyber,
  title   = {TensorKyber: Reformulating the NTT Implementation on Tensor Cores
             for High-Throughput Performance in ML-KEM},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## License

Released for research and reproducibility. See the repository root for the license terms;
the bundled reference C implementation under `ref/` is derived from the public
ML-KEM / Kyber reference code.



