// ============================================================================
//  Standalone NTT / iNTT benchmark -- Kyber (N=256, q=3329)
//
//  Self-contained port of the original bench_ntt.cu that keeps ONLY these four
//  batch paths (32 tasks/warp, one polynomial per task):
//
//    1) 2TC      -- register-staged, WMMA per-block Stage-2.  Each lane holds
//                   its own polynomial and the 16-slot shared buffer is staged
//                   out of registers.
//    2) TC-CC    -- register-staged, scalar-butterfly Stage-2.  Same staging.
//    3) TC-CC-CP -- cp.async single-buffer staging, scalar-butterfly Stage-2.
//    4) TC-PACK  -- packed 2-bit ternary forward (4 coeffs/byte, 64 B/poly)
//                   cp.async'd into shared, unpacked into the standard int16
//                   slot layout, forward core run with the BASIC (single-limb)
//                   M1 and scalar Stage-2.  Its inverse is TC-CC-CP.
//
//  Usage:  ./ntt_bench [npoly] [runs]        (defaults: 262144 30)
// ============================================================================
#include "ntt_kernels.cuh"
#include <algorithm>

// ---- reference host transforms (from ref/, for byte-exact validation) -----
// The namespace must match the -DKYBER_K given to the ref objects.
#if   KYBER_K == 2
extern "C" void pqcrystals_kyber512_ref_ntt(int16_t*);
extern "C" void pqcrystals_kyber512_ref_invntt(int16_t*);
#define REF_NTT     pqcrystals_kyber512_ref_ntt
#define REF_INVNTT  pqcrystals_kyber512_ref_invntt
#elif KYBER_K == 3
extern "C" void pqcrystals_kyber768_ref_ntt(int16_t*);
extern "C" void pqcrystals_kyber768_ref_invntt(int16_t*);
#define REF_NTT     pqcrystals_kyber768_ref_ntt
#define REF_INVNTT  pqcrystals_kyber768_ref_invntt
#else
extern "C" void pqcrystals_kyber1024_ref_ntt(int16_t*);
extern "C" void pqcrystals_kyber1024_ref_invntt(int16_t*);
#define REF_NTT     pqcrystals_kyber1024_ref_ntt
#define REF_INVNTT  pqcrystals_kyber1024_ref_invntt
#endif

// ============================================================================
//    S2 = 1 -> "2TC"    layers 5..7 folded into per-block WMMA tiles
//    S2 = 0 -> "TC-CC"  layers 5..7 as scalar butterflies
// ============================================================================
#define BATCH_BUF_BYTES (16*TC_SLOT_STRIDE*sizeof(int16_t))

// Per-warp context for the register-staged kernels.  s_blo / s_bhi / s_acc are
// not used by the register-staged wrapper and are aliased onto the polynomial
// buffer so the wrapper keeps a single call signature.
__device__ __forceinline__ void reg_stage_ctx(uint8_t* smem, int w,
                                              int16_t*& s_polys, int8_t*& s_blo,
                                              int8_t*& s_bhi, int32_t*& s_acc)
{
  uint8_t* base = smem + TC_MAT_BYTES + (size_t)w*BATCH_BUF_BYTES;
  s_polys = (int16_t*)base;
  s_blo   = (int8_t*)s_polys;
  s_bhi   = (int8_t*)s_polys;
  s_acc   = (int32_t*)s_polys;
}

template<int S2>
__global__ void k_ntt_2tc_tc_cc(int16_t* __restrict__ d, int npoly)
{
  extern __shared__ uint8_t smem[];
  const int w    = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wpb  = blockDim.x >> 5;

  stage_tc_matrices(smem);                    // block-wide __syncthreads()

  const int64_t id = (int64_t)((size_t)blockIdx.x * wpb + w) * 32 + lane;
  if (id >= npoly) {
    return;
  }

  __align__(16) int16_t poly[1][KYBER_N];
  #pragma unroll
  for (int w4 = 0; w4 < KYBER_N/8; w4++) {
    ((int4*)poly[0])[w4] = ((const int4*)(d + id*KYBER_N))[w4];
  }

  const int8_t* M = (const int8_t*)smem;      // [M1_lo | M1_hi | Minv_lo | Minv_hi]
  int16_t* s_polys;
  int8_t*  s_blo;
  int8_t*  s_bhi;
  int32_t* s_acc;
  reg_stage_ctx(smem, w, s_polys, s_blo, s_bhi, s_acc);

  ntt_regstage<false, S2>(poly, 1, M, M+256, s_polys, s_blo, s_bhi, s_acc);

  #pragma unroll
  for (int w4 = 0; w4 < KYBER_N/8; w4++) {
    ((int4*)(d + id*KYBER_N))[w4] = ((const int4*)poly[0])[w4];
  }
}

template<int S2>
__global__ void k_invntt_2tc_tc_cc(int16_t* __restrict__ d, int npoly)
{
  extern __shared__ uint8_t smem[];
  const int w    = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wpb  = blockDim.x >> 5;

  stage_tc_matrices(smem);

  const int64_t id = (int64_t)((size_t)blockIdx.x * wpb + w) * 32 + lane;
  if (id >= npoly) {
    return;
  }

  __align__(16) int16_t poly[1][KYBER_N];
  #pragma unroll
  for (int w4 = 0; w4 < KYBER_N/8; w4++) {
    ((int4*)poly[0])[w4] = ((const int4*)(d + id*KYBER_N))[w4];
  }

  const int8_t* M = (const int8_t*)smem;
  int16_t* s_polys;
  int8_t*  s_blo;
  int8_t*  s_bhi;
  int32_t* s_acc;
  reg_stage_ctx(smem, w, s_polys, s_blo, s_bhi, s_acc);

  intt_regstage<S2>(poly, 1, M+512, M+768, s_polys, s_blo, s_bhi, s_acc);

  #pragma unroll
  for (int w4 = 0; w4 < KYBER_N/8; w4++) {
    ((int4*)(d + id*KYBER_N))[w4] = ((const int4*)poly[0])[w4];
  }
}



// Un-elidable warp barrier.  __syncwarp() can be dropped by the compiler when
// the code is warp-uniform, but cp.async writes issued by OTHER lanes only
// become visible to the whole warp after a real bar.warp.sync (DEPBAR alone is
// per-thread).  Used by both cp.async staged kernel families.
__device__ __forceinline__ void cp_warp_barrier()
{
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
}

// cp.async the 16 polynomials of round r (tasks warpbase + 16r .. +15) from
// global memory into slots b*16 .. b*16+15 of s_buf.  All 32 lanes call this;
// it commits one pipeline group.  Tasks beyond the batch are skipped.
__device__ __forceinline__ void cp_stage_round(const int16_t* __restrict__ d,
                                              int64_t warpbase, int r, int b,
                                              int16_t* s_buf, int npoly)
{
  const int lane = threadIdx.x & 31;
  const int64_t base = warpbase + (int64_t)r*16;
  const bool tail = (base + 15 >= npoly);   // the round reaches past the batch

  #pragma unroll
  for (int k = 0; k < 16; k++) {
    const int c   = 16*lane + k;            // 0..511: 16-byte chunk index
    const int p   = c >> 5;                 // polynomial within the round
    const int sub = c & 31;                 // 16-byte offset within the poly
    const int64_t task = base + p;
    if (!tail || task < npoly) {
      __pipeline_memcpy_async(
          (void*)(s_buf + (size_t)b*16*TC_SLOT_STRIDE + (size_t)p*TC_SLOT_STRIDE + sub*8),
          (const void*)(d + task*KYBER_N + sub*8), 16);
    }
  }
  __pipeline_commit();
}


// ============================================================================
//  TC-CC-CP: cp.async staged batch kernels, 32 tasks per warp, one polynomial
//  per task.  ONE 16-slot buffer per warp, filled straight from global memory
//  with cp.async (no per-lane input register array, no double buffering);
//  round r+1 is staged only after round r has been read back.  Stage-2 is
//  scalar.  Same shared layout as 2TC / TC-CC.
//
//  A round covers 16 polynomials: in round 0 lanes 0..15 own slots 0..15 and
//  read their own result back, in round 1 lanes 16..31 do.
// ============================================================================
template<int S2>
__global__ void k_ntt_tc_cc_cp(int16_t* __restrict__ d, int npoly)
{
  extern __shared__ uint8_t smem[];
  const int w    = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wpb  = blockDim.x >> 5;

  stage_tc_matrices(smem);                    // block-wide __syncthreads()

  const int64_t warpbase = (int64_t)((size_t)blockIdx.x * wpb + w) * 32;
  if (warpbase >= npoly) {
    return;
  }
  const int8_t* M = (const int8_t*)smem;
  int16_t* s_buf = (int16_t*)(smem + TC_MAT_BYTES) + (size_t)w*16*TC_SLOT_STRIDE;
  const int64_t id = warpbase + lane;
  const bool alive = (id < npoly);

  // ---- round 0: tasks warpbase+0 .. +15 ----------------------------------
  cp_stage_round(d, warpbase, 0, 0, s_buf, npoly);
  __pipeline_wait_prior(0);
  cp_warp_barrier();
  ntt_core<false, S2>(s_buf, M, M+256);
  if (alive && (lane < 16)) {
    const int4* src = (const int4*)(s_buf + (size_t)lane*TC_SLOT_STRIDE);
    int4*       dst = (int4*)(d + id*KYBER_N);
    #pragma unroll
    for (int w4 = 0; w4 < KYBER_N/8; w4++) {
      dst[w4] = src[w4];
    }
  }
  cp_warp_barrier();                          // round-0 read-back complete

  // ---- round 1: tasks warpbase+16 .. +31 ---------------------------------
  cp_stage_round(d, warpbase, 1, 0, s_buf, npoly);
  __pipeline_wait_prior(0);
  cp_warp_barrier();
  ntt_core<false, S2>(s_buf, M, M+256);
  if (alive && (lane >= 16)) {
    const int4* src = (const int4*)(s_buf + (size_t)(lane-16)*TC_SLOT_STRIDE);
    int4*       dst = (int4*)(d + id*KYBER_N);
    #pragma unroll
    for (int w4 = 0; w4 < KYBER_N/8; w4++) {
      dst[w4] = src[w4];
    }
  }
}

template<int S2>
__global__ void k_invntt_tc_cc_cp(int16_t* __restrict__ d, int npoly)
{
  extern __shared__ uint8_t smem[];
  const int w    = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wpb  = blockDim.x >> 5;

  stage_tc_matrices(smem);

  const int64_t warpbase = (int64_t)((size_t)blockIdx.x * wpb + w) * 32;
  if (warpbase >= npoly) {
    return;
  }
  const int8_t* M = (const int8_t*)smem;      // Minv lives at M+512 / M+768
  int16_t* s_buf = (int16_t*)(smem + TC_MAT_BYTES) + (size_t)w*16*TC_SLOT_STRIDE;
  const int64_t id = warpbase + lane;
  const bool alive = (id < npoly);

  // ---- round 0 -----------------------------------------------------------
  cp_stage_round(d, warpbase, 0, 0, s_buf, npoly);
  __pipeline_wait_prior(0);
  cp_warp_barrier();
  intt_core<S2>(s_buf, M+512, M+768);
  if (alive && (lane < 16)) {
    const int4* src = (const int4*)(s_buf + (size_t)lane*TC_SLOT_STRIDE);
    int4*       dst = (int4*)(d + id*KYBER_N);
    #pragma unroll
    for (int w4 = 0; w4 < KYBER_N/8; w4++) {
      dst[w4] = src[w4];
    }
  }
  cp_warp_barrier();

  // ---- round 1 -----------------------------------------------------------
  cp_stage_round(d, warpbase, 1, 0, s_buf, npoly);
  __pipeline_wait_prior(0);
  cp_warp_barrier();
  intt_core<S2>(s_buf, M+512, M+768);
  if (alive && (lane >= 16)) {
    const int4* src = (const int4*)(s_buf + (size_t)(lane-16)*TC_SLOT_STRIDE);
    int4*       dst = (int4*)(d + id*KYBER_N);
    #pragma unroll
    for (int w4 = 0; w4 < KYBER_N/8; w4++) {
      dst[w4] = src[w4];
    }
  }
}
// ============================================================================
// ============================================================================
//  TC-PACK: packed-input batch forward NTT, 32 tasks per warp.
//  Only the forward is packed (the inverse input is the full 12-bit NTT domain),
//  so the benchmark pairs TC-PACK with the TC-CC-CP inverse.
// ============================================================================
// packed bytes per polynomial, rounded up to a whole number of 16-byte chunks
#define PACK_POLY_BYTES(PB) (16*((KYBER_N + (8/(PB)) - 1)/(8/(PB))))
// packed buffer plus the 16-slot int16 buffer that the compute core reads
#define PACK_WARP_BYTES(PB) (PACK_POLY_BYTES(PB) + 16*TC_SLOT_STRIDE*sizeof(int16_t))

// cp.async the packed form of round r's 16 polynomials into s_packed.
template<int PB>
__device__ __forceinline__ void pack_stage(const int8_t* __restrict__ d_in,
                                          int64_t warpbase, int r,
                                          int8_t* s_packed, int npoly)
{
  const int lane = threadIdx.x & 31;
  constexpr int PER    = 8/PB;                    // coefficients per byte
  constexpr int PPB    = (KYBER_N + PER - 1)/PER; // packed bytes per poly
  constexpr int CPOL   = PPB/16;                  // 16-byte chunks per poly
  constexpr int CHUNKS = 16*CPOL;                 // 16-byte chunks per round
  const int64_t base = warpbase + (int64_t)r*16;
  const bool tail = (base + 15 >= npoly);

  #pragma unroll
  for (int k = 0; k < CHUNKS/32; k++) {
    const int c   = (CHUNKS/32)*lane + k;         // chunk index
    const int p   = c / CPOL;                     // poly within the round
    const int sub = c % CPOL;                     // chunk within the poly
    const int64_t task = base + p;
    if (!tail || task < npoly) {
      __pipeline_memcpy_async((void*)(s_packed + (size_t)p*PPB + sub*16),
                              (const void*)(d_in + task*PPB + sub*16), 16);
    }
  }
  __pipeline_commit();
}

// unpack 16 packed polynomials (PPB bytes each) into 16 int16 slots.
template<int PB>
__device__ __forceinline__ void pack_unpack(const int8_t* s_packed, int16_t* s_polys)
{
  const int lane = threadIdx.x & 31;
  constexpr int PER  = 8/PB;                      // coefficients per byte
  constexpr int PPB  = (KYBER_N + PER - 1)/PER;   // packed bytes per poly
  constexpr int BYTES_PER_LANE = 16*PPB/32;

  #pragma unroll
  for (int b = 0; b < BYTES_PER_LANE; b++) {
    const int gb = BYTES_PER_LANE*lane + b;
    const int p  = gb / PPB;
    const int pb = gb % PPB;
    const uint8_t byte = (uint8_t)s_packed[gb];

    #pragma unroll
    for (int f = 0; f < PER; f++) {
      const int coeff = pb*PER + f;
      const int code  = (byte >> (f*PB)) & ((1<<PB)-1);
      // sign-extend the PB-bit two's-complement field: cast to int16 BEFORE the
      // arithmetic right shift, otherwise bit 15 is not the sign.
      const int16_t v = (int16_t)((int16_t)(code << (16-PB)) >> (16-PB));
      s_polys[p*TC_SLOT_STRIDE + coeff] = v;
    }
  }
}

// TC-PACK forward.  Same 32-tasks-per-warp geometry as TC-CC-CP, but the input
// is the packed buffer and the core runs the single-limb BASIC M1.
template<int PB, int S2>
__global__ void k_ntt_tc_pack(const int8_t* __restrict__ d_in,
                              int16_t* __restrict__ d_out, int npoly)
{
  extern __shared__ uint8_t smem[];
  const int w    = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wpb  = blockDim.x >> 5;

  stage_tc_matrices(smem);

  const int64_t warpbase = (int64_t)((size_t)blockIdx.x * wpb + w) * 32;
  if (warpbase >= npoly) {
    return;
  }
  const int8_t* M = (const int8_t*)smem;
  int8_t*  s_packed = (int8_t*)(smem + TC_MAT_BYTES) + (size_t)w*PACK_WARP_BYTES(PB);
  int16_t* s_polys  = (int16_t*)(s_packed + PACK_POLY_BYTES(PB));
  const int64_t id = warpbase + lane;
  const bool alive = (id < npoly);

  for (int round = 0; round < 2; round++) {
    const int slot = lane - round*16;

    pack_stage<PB>(d_in, warpbase, round, s_packed, npoly);
    __pipeline_wait_prior(0);
    cp_warp_barrier();

    pack_unpack<PB>(s_packed, s_polys);
    cp_warp_barrier();

    ntt_core<true, S2>(s_polys, M, M+256);

    // lanes 0..15 own slots 0..15 of round 0, lanes 16..31 of round 1
    if (alive && (unsigned)slot < 16u) {
      const int4* src = (const int4*)(s_polys + (size_t)slot*TC_SLOT_STRIDE);
      int4*       dst = (int4*)(d_out + id*KYBER_N);
      #pragma unroll
      for (int w4 = 0; w4 < KYBER_N/8; w4++) {
        dst[w4] = src[w4];
      }
    }
    cp_warp_barrier();
  }
}

// ============================================================================
//  Host-side helpers
// ============================================================================

// normalise a coefficient to [0,q) so two ranges can be compared mod q
static inline int16_t norm_q(int16_t v)
{
  int t = v % KYBER_Q;
  if (t < 0) {
    t += KYBER_Q;
  }
  return (int16_t)t;
}

// number of coefficients that differ mod q between the first `chk` polynomials
// of `got` and of the reference `want`
static int count_mismatches(const int16_t* got, const int16_t* want, int chk)
{
  int bad = 0;
  for (int i = 0; i < chk; i++) {
    for (int j = 0; j < KYBER_N; j++) {
      const size_t idx = (size_t)i*KYBER_N + j;
      if (norm_q(got[idx]) != norm_q(want[idx])) {
        bad++;
      }
    }
  }
  return bad;
}

// ---------------------------------------------------------------------------
// run statistics over the timed runs
// ---------------------------------------------------------------------------
#define MAX_RUNS 64

static double medianof(const double* v, int n)
{
  double t[MAX_RUNS];
  for (int i = 0; i < n; i++) {
    t[i] = v[i];
  }
  std::sort(t, t + n);
  return (n & 1) ? t[n/2] : 0.5 * (t[n/2 - 1] + t[n/2]);
}

static double stdev_of(const double* v, int n, double c)
{
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    const double d = v[i] - c;
    s += d * d;
  }
  return (n > 1) ? sqrt(s / (n - 1)) : 0.0;
}

// mean / median / sample std-dev of one run vector
struct Stat {
  double mean;
  double med;
  double sd;
};

static Stat statof(const double* v, int n)
{
  Stat s;
  double m = 0.0;
  for (int i = 0; i < n; i++) {
    m += v[i];
  }
  m /= n;

  s.mean = m;
  s.med  = medianof(v, n);
  s.sd   = stdev_of(v, n, m);
  return s;
}

// per-run latency vector (ms) -> throughput statistics (poly/s)
static Stat throughput_of(const double* ms, int n, int npoly)
{
  double tp[MAX_RUNS];
  for (int i = 0; i < n; i++) {
    tp[i] = (double)npoly / (ms[i] / 1000.0);
  }
  return statof(tp, n);
}

// forward + inverse back to back -> round-trip throughput statistics (poly/s)
static Stat roundtrip_of(const double* ms_f, const double* ms_i, int n, int npoly)
{
  double tp[MAX_RUNS];
  for (int i = 0; i < n; i++) {
    tp[i] = (double)npoly / ((ms_f[i] + ms_i[i]) / 1000.0);
  }
  return statof(tp, n);
}

static void print_stat_line(const char* what, const Stat& s, double lat_ms)
{
  printf("  %-20s mean %12.0f  median %12.0f  +/- %-9.0f poly/s   lat %8.4f ms\n",
         what, s.mean, s.med, s.sd, lat_ms);
}

static void print_roundtrip_line(const Stat& s)
{
  printf("  Round-trip (batch)      mean %12.0f  median %12.0f poly/s\n\n",
         s.mean, s.med);
}

// ============================================================================
//  Summary table: every benchmark section records one row, main() prints the
//  consolidated table at the end.  Each row carries the mean and the median.
// ============================================================================
#define MAX_BENCH_ROWS 8

struct BenchRow {
  char        name[48];   // short variant label ("2TC", "TC-PACK", ...)
  const char* geometry;   // e.g. "32 tasks/warp"
  int         block;      // block size
  Stat        ntt;
  Stat        intt;
  Stat        rt;         // NTT + iNTT round trip
  int         ntt_bad;    // mismatches vs the reference (0 = byte-exact)
  int         intt_bad;
};

struct BenchRows {
  BenchRow row[MAX_BENCH_ROWS];
  int      n;
};

static void bench_add(BenchRows& R, const char* name, const char* geometry, int block,
                      const Stat& ntt, const Stat& intt, const Stat& rt,
                      int ntt_bad, int intt_bad)
{
  if (R.n >= MAX_BENCH_ROWS) {
    return;
  }
  BenchRow& x = R.row[R.n++];

  snprintf(x.name, sizeof x.name, "%s", name);
  x.geometry = geometry;
  x.block    = block;
  x.ntt      = ntt;
  x.intt     = intt;
  x.rt       = rt;
  x.ntt_bad  = ntt_bad;
  x.intt_bad = intt_bad;
}

static const char* ok_str(int bad)
{
  return bad ? "FAIL" : "ok";
}

static void bench_print_summary(const BenchRows& R, int npoly, int runs)
{
  printf("\n==========================================================================================\n");
  printf(" SUMMARY: NTT/iNTT batch variants -- mean & median over %d runs, %d polys (M poly/s)\n",
         runs, npoly);
  printf("==========================================================================================\n");
  printf("%-30s %-12s %5s |%10s %10s |%10s %10s |%10s %10s | %5s %5s\n",
         "Variant", "geometry", "block",
         "NTT mean", "NTT med", "iNTT mean", "iNTT med", "RT mean", "RT med",
         "NTT", "iNTT");
  printf("%-30s %-12s %5s |%10s %10s |%10s %10s |%10s %10s | %5s %5s\n",
         "-", "-", "-", "M/s", "M/s", "M/s", "M/s", "M/s", "M/s", "ok", "ok");

  for (int i = 0; i < R.n; i++) {
    const BenchRow& x = R.row[i];
    printf("%-30s %-12s %5d |%10.1f %10.1f |%10.1f %10.1f |%10.1f %10.1f | %5s %5s\n",
           x.name, x.geometry, x.block,
           x.ntt.mean/1e6,  x.ntt.med/1e6,
           x.intt.mean/1e6, x.intt.med/1e6,
           x.rt.mean/1e6,   x.rt.med/1e6,
           ok_str(x.ntt_bad), ok_str(x.intt_bad));
  }
  printf("==========================================================================================\n");
}

// Print one variant's result section and record its summary row.  `label` names
// the forward kernel, `inv_label` the inverse kernel it is paired with.
// Returns the number of mismatching coefficients (0 = byte-exact).
static int bench_report(BenchRows& R, const char* label, const char* inv_label,
                        int block, int npoly, int RUNS,
                        const double* ms_f, const double* ms_i, int e_f, int e_i)
{
  const Stat ntts  = throughput_of(ms_f, RUNS, npoly);
  const Stat intts = throughput_of(ms_i, RUNS, npoly);
  const Stat rts   = roundtrip_of(ms_f, ms_i, RUNS, npoly);

  printf("\n--- Batch path (32 tasks/warp, %s NTT / %s iNTT, block=%d) ---\n",
         label, inv_label, block);
  print_stat_line("NTT  batch", ntts,  medianof(ms_f, RUNS));
  print_stat_line("iNTT batch", intts, medianof(ms_i, RUNS));
  print_roundtrip_line(rts);

  bench_add(R, label, "32 tasks/warp", block, ntts, intts, rts, e_f, e_i);
  return e_f + e_i;
}

// ============================================================================
//  2TC / TC-CC benchmark -- the register-staged variants.
//    S2 = 1 -> "2TC"    (layers 5..7 folded into per-block WMMA tiles)
//    S2 = 0 -> "TC-CC"  (layers 5..7 as scalar butterflies)
//  Shared footprint: TC_MAT_BYTES + one 16-slot buffer per warp.
// ============================================================================
template<int S2>
static int bench_2tc_tc_cc(int block, int npoly, int RUNS,
                           int16_t* h_in, int16_t* h_ref_ntt, int16_t* h_ref_inv,
                           int16_t* h_out, int16_t* d_buf, size_t bytes, BenchRows& R)
{
  const int wpb    = block / 32;
  const int grid   = (int)(((size_t)npoly + (size_t)wpb*32 - 1) / ((size_t)wpb*32));
  const size_t shm = TC_MAT_BYTES + (size_t)wpb*BATCH_BUF_BYTES;
  const int CHK    = (npoly < 64) ? npoly : 64;
  const char* LABEL = S2 ? "2TC" : "TC-CC";

  // ---- correctness -------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
  k_ntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_f = count_mismatches(h_out, h_ref_ntt, CHK);

  k_invntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_i = count_mismatches(h_out, h_ref_inv, CHK);

  printf("  %-20s NTT %s (%d bad)   iNTT %s (%d bad)\n",
         LABEL, ok_str(e_f), e_f, ok_str(e_i), e_i);
  if (e_f + e_i) {
    printf("  !! %s mismatches present -- timings not trustworthy\n", LABEL);
  }

  // ---- timing ------------------------------------------------------------
  double ms_f[MAX_RUNS];
  double ms_i[MAX_RUNS];
  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  float ms;

  CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
  for (int w = 0; w < 2; w++) {                        // warm-up
    k_ntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
    k_invntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < RUNS; r++) {
    // re-stage the input so every run starts from the same state
    CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEventRecord(ev0);
    k_ntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_f[r] = ms;

    cudaEventRecord(ev0);
    k_invntt_2tc_tc_cc<S2><<<grid,block,shm>>>(d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_i[r] = ms;
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  return bench_report(R, LABEL, LABEL, block, npoly, RUNS, ms_f, ms_i, e_f, e_i);
}


// ============================================================================
//  TC-CC-CP benchmark: cp.async staging straight from global memory into ONE
//  16-slot shared buffer per warp, scalar Stage-2.
// ============================================================================
static int bench_tc_cc_cp(int block, int npoly, int RUNS,
                          int16_t* h_in, int16_t* h_ref_ntt, int16_t* h_ref_inv,
                          int16_t* h_out, int16_t* d_buf, size_t bytes, BenchRows& R)
{
  const int wpb    = block / 32;
  const int grid   = (int)(((size_t)npoly + (size_t)wpb*32 - 1) / ((size_t)wpb*32));
  const size_t shm = TC_MAT_BYTES + (size_t)wpb*BATCH_BUF_BYTES;
  const int CHK    = (npoly < 64) ? npoly : 64;
  const char* LABEL = "TC-CC-CP";

  // ---- correctness -------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
  k_ntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_f = count_mismatches(h_out, h_ref_ntt, CHK);

  k_invntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_i = count_mismatches(h_out, h_ref_inv, CHK);

  printf("  %-20s NTT %s (%d bad)   iNTT %s (%d bad)\n",
         LABEL, ok_str(e_f), e_f, ok_str(e_i), e_i);
  if (e_f + e_i) {
    printf("  !! %s mismatches present -- timings not trustworthy\n", LABEL);
  }

  // ---- timing ------------------------------------------------------------
  double ms_f[MAX_RUNS];
  double ms_i[MAX_RUNS];
  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  float ms;

  CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
  for (int w = 0; w < 2; w++) {                        // warm-up
    k_ntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
    k_invntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < RUNS; r++) {
    // re-stage the input so every run starts from the same state
    CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEventRecord(ev0);
    k_ntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_f[r] = ms;

    cudaEventRecord(ev0);
    k_invntt_tc_cc_cp<0><<<grid,block,shm>>>(d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_i[r] = ms;
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  return bench_report(R, LABEL, LABEL, block, npoly, RUNS, ms_f, ms_i, e_f, e_i);
}
// ============================================================================
//  TC-PACK benchmark.  Generates a small-coefficient input (PB bits: ternary
//  for PB=2, CBD-like [-2,2] for PB=3), packs it, runs the packed forward NTT
//  (TC-PACK) followed by the standard TC-CC-CP inverse, validates both
//  byte-exact against the reference, and also times the TC-CC-CP forward on the
//  SAME small input (as int16) so the packing benefit is a head-to-head on
//  identical data.
// ============================================================================
// host-side: pack one polynomial of PB-bit small coefficients
// (two's-complement fields, little-endian within each byte)
static void pack_small(int bits, const int16_t* poly, uint8_t* packed)
{
  const int per = 8 / bits;
  memset(packed, 0, (KYBER_N + per - 1) / per);
  for (int i = 0; i < KYBER_N; i++) {
    const int v = poly[i];
    packed[i/per] |= (uint8_t)(v & ((1<<bits)-1)) << ((i%per)*bits);
  }
}

template<int PB>
static int bench_tc_pack(int block, int npoly, int RUNS,
                         int16_t* d_buf, size_t bytes, BenchRows& R)
{
  const int wpb  = block / 32;
  const int grid = (int)(((size_t)npoly + (size_t)wpb*32 - 1) / ((size_t)wpb*32));
  constexpr int PER = 8 / PB;                     // coefficients per byte
  constexpr int PPB = (KYBER_N + PER - 1)/PER;    // packed bytes per polynomial
  const size_t packed_bytes = (size_t)npoly*PPB;
  const size_t shm    = TC_MAT_BYTES + (size_t)wpb*PACK_WARP_BYTES(PB);
  const size_t cp_shm = TC_MAT_BYTES + (size_t)wpb*BATCH_BUF_BYTES;
  const int CHK = (npoly < 64) ? npoly : 64;
  // only the 2-bit ternary instantiation is benchmarked; PB stays a kernel knob
  const char* LABEL = (PB == 2) ? "TC-PACK" : "TC-PACK (cbd 2/B)";
  const int RNG = (PB == 2) ? 3 : 5;              // {-1,0,1} or {-2..2}

  // ---- small-coefficient input, packed form and host references ----------
  int16_t* h_in      = (int16_t*)malloc(bytes);
  int16_t* h_out     = (int16_t*)malloc(bytes);
  int16_t* h_ref_ntt = (int16_t*)malloc(bytes);
  int16_t* h_ref_inv = (int16_t*)malloc(bytes);
  uint8_t* h_packed  = (uint8_t*)malloc(packed_bytes);

  srand(20240819 + PB*10);
  for (size_t i = 0; i < (size_t)npoly*KYBER_N; i++) {
    h_in[i] = (int16_t)(rand() % RNG - RNG/2);
  }
  for (int i = 0; i < CHK; i++) {
    int16_t t[KYBER_N];
    memcpy(t, h_in + (size_t)i*KYBER_N, sizeof(t));
    REF_NTT(t);
    memcpy(h_ref_ntt + (size_t)i*KYBER_N, t, sizeof(t));
    REF_INVNTT(t);
    memcpy(h_ref_inv + (size_t)i*KYBER_N, t, sizeof(t));
  }
  for (int i = 0; i < npoly; i++) {
    pack_small(PB, h_in + (size_t)i*KYBER_N, h_packed + (size_t)i*PPB);
  }

  int8_t* d_packed;
  CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));

  // ---- correctness: packed forward, then the TC-CC-CP inverse ------------
  CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_bytes, cudaMemcpyHostToDevice));
  k_ntt_tc_pack<PB,0><<<grid,block,shm>>>(d_packed, d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_f = count_mismatches(h_out, h_ref_ntt, CHK);

  k_invntt_tc_cc_cp<0><<<grid,block,cp_shm>>>(d_buf, npoly);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out, d_buf, bytes, cudaMemcpyDeviceToHost));
  const int e_i = count_mismatches(h_out, h_ref_inv, CHK);

  printf("  %-20s NTT %s (%d bad)   iNTT %s (%d bad)\n",
         LABEL, ok_str(e_f), e_f, ok_str(e_i), e_i);
  if (e_f + e_i) {
    printf("  !! %s mismatches present -- timings not trustworthy\n", LABEL);
  }

  // ---- timing: packed forward + TC-CC-CP inverse -------------------------
  double ms_f[MAX_RUNS];
  double ms_i[MAX_RUNS];
  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  float ms;

  for (int w = 0; w < 2; w++) {                        // warm-up
    CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_bytes, cudaMemcpyHostToDevice));
    k_ntt_tc_pack<PB,0><<<grid,block,shm>>>(d_packed, d_buf, npoly);
    k_invntt_tc_cc_cp<0><<<grid,block,cp_shm>>>(d_buf, npoly);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < RUNS; r++) {
    CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_bytes, cudaMemcpyHostToDevice));

    cudaEventRecord(ev0);
    k_ntt_tc_pack<PB,0><<<grid,block,shm>>>(d_packed, d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_f[r] = ms;

    cudaEventRecord(ev0);
    k_invntt_tc_cc_cp<0><<<grid,block,cp_shm>>>(d_buf, npoly);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms_i[r] = ms;
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  const Stat ntts  = throughput_of(ms_f, RUNS, npoly);
  const Stat intts = throughput_of(ms_i, RUNS, npoly);
  const Stat rts   = roundtrip_of(ms_f, ms_i, RUNS, npoly);

  printf("\n--- Batch path (32 tasks/warp, %s NTT / TC-CC-CP iNTT, block=%d) ---\n",
         LABEL, block);
  print_stat_line("NTT  batch (packed)", ntts,  medianof(ms_f, RUNS));
  print_stat_line("iNTT batch (1buf)",   intts, medianof(ms_i, RUNS));
  print_roundtrip_line(rts);

  // ---- head-to-head: TC-CC-CP forward on the SAME small input ------------
  {
    double ms_cp[MAX_RUNS];
    CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
    for (int w = 0; w < 2; w++) {                      // warm-up
      k_ntt_tc_cc_cp<0><<<grid,block,cp_shm>>>(d_buf, npoly);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < RUNS; r++) {
      CUDA_CHECK(cudaMemcpy(d_buf, h_in, bytes, cudaMemcpyHostToDevice));
      cudaEventRecord(e0);
      k_ntt_tc_cc_cp<0><<<grid,block,cp_shm>>>(d_buf, npoly);
      cudaEventRecord(e1);
      CUDA_CHECK(cudaEventSynchronize(e1));
      float el;
      cudaEventElapsedTime(&el, e0, e1);
      ms_cp[r] = el;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    const Stat cps = throughput_of(ms_cp, RUNS, npoly);
    printf("  [same data] TC-CC-CP fwd (int16): %14.0f poly/s   (packing+basic gain = %.1f%%)\n\n",
           cps.med, (ntts.med / cps.med - 1.0) * 100.0);
  }

  bench_add(R, LABEL, "32 tasks/warp", block, ntts, intts, rts, e_f, e_i);

  free(h_in);
  free(h_out);
  free(h_ref_ntt);
  free(h_ref_inv);
  free(h_packed);
  CUDA_CHECK(cudaFree(d_packed));
  return e_f + e_i;
}
// ============================================================================
//  main: standalone NTT/iNTT benchmark.
//  Defaults: 262144 polynomials in the batch and 30 timed runs; every variant
//  is reported with both the mean and the median throughput.
// ============================================================================
int main(int argc, char** argv)
{
  int npoly = (argc > 1) ? atoi(argv[1]) : 262144;   // batch size (polynomials)
  int RUNS  = (argc > 2) ? atoi(argv[2]) : 30;       // timed runs
  if (RUNS > MAX_RUNS) {
    RUNS = MAX_RUNS;
  }
  if (RUNS < 1) {
    RUNS = 1;
  }

  const char* PARAM = (KYBER_K == 2) ? "Kyber-512" :
                      (KYBER_K == 3) ? "Kyber-768" : "Kyber-1024";

  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("==============================================================================\n");
  printf(" Standalone NTT / iNTT benchmark -- %s (N=256, q=3329)\n", PARAM);
  printf(" GPU: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  printf(" batch = %d polynomials, %d timed runs (mean & median reported)\n", npoly, RUNS);
  printf(" variants: 2TC | TC-CC | TC-CC-CP | TC-PACK\n");
  printf("==============================================================================\n");

  CUDA_CHECK(cudaMemcpyToSymbol(d_zetas, zetas_host, sizeof(zetas_host)));
  setup_tc_matrices();

  const size_t bytes = (size_t)npoly*KYBER_N*sizeof(int16_t);
  int16_t* h_in      = (int16_t*)malloc(bytes);
  int16_t* h_out     = (int16_t*)malloc(bytes);
  int16_t* h_ref_ntt = (int16_t*)malloc(bytes);
  int16_t* h_ref_inv = (int16_t*)malloc(bytes);

  // random input in the Kyber coefficient range [0,q)
  srand(20240816);
  for (size_t i = 0; i < (size_t)npoly*KYBER_N; i++) {
    h_in[i] = (int16_t)(rand() % KYBER_Q);
  }

  // reference NTT and iNTT(NTT(x)) on the host for the first CHK polynomials
  const int CHK = (npoly < 64) ? npoly : 64;
  for (int i = 0; i < CHK; i++) {
    int16_t t[KYBER_N];
    memcpy(t, h_in + (size_t)i*KYBER_N, sizeof(t));
    REF_NTT(t);
    memcpy(h_ref_ntt + (size_t)i*KYBER_N, t, sizeof(t));
    REF_INVNTT(t);
    memcpy(h_ref_inv + (size_t)i*KYBER_N, t, sizeof(t));
  }

  int16_t* d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, bytes));

  printf("\n--- Correctness vs reference C (first %d polys, mod-q normalised) ---\n", CHK);
  BenchRows R;
  R.n = 0;
  int allbad = 0;

  // ---- 2TC (S2=1) and TC-CC (S2=0): register-staged, block=64 ------------
  allbad += bench_2tc_tc_cc<1>(64, npoly, RUNS, h_in, h_ref_ntt, h_ref_inv, h_out, d_buf, bytes, R);
  allbad += bench_2tc_tc_cc<0>(64, npoly, RUNS, h_in, h_ref_ntt, h_ref_inv, h_out, d_buf, bytes, R);

  // ---- TC-CC-CP: cp.async single-buffer, block=64 ------------------------
  allbad += bench_tc_cc_cp(64, npoly, RUNS, h_in, h_ref_ntt, h_ref_inv, h_out, d_buf, bytes, R);

  // ---- TC-PACK: packed ternary forward, scalar Stage-2, block=64 ---------
  allbad += bench_tc_pack<2>(64, npoly, RUNS, d_buf, bytes, R);

  if (allbad) {
    printf("\n  !! %d mismatches present -- timings above are NOT trustworthy\n", allbad);
  } else {
    printf("\n  all variants byte-exact vs the reference C NTT/iNTT\n");
  }

  // ---- consolidated summary table ---------------------------------------
  bench_print_summary(R, npoly, RUNS);

  cudaFree(d_buf);
  free(h_in);
  free(h_out);
  free(h_ref_ntt);
  free(h_ref_inv);
  return allbad ? 1 : 0;
}
