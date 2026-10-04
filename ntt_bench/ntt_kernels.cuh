// ============================================================================
//  Tensor-core NTT/iNTT device kernels -- Kyber-512 (N=256, q=3329)
//
//  Everything the standalone NTT/iNTT benchmark needs, and nothing else:
//
//    * mod-q reductions (Montgomery / Barrett / unsigned Barrett)
//    * the twiddle table (zetas_host host copy + d_zetas device copy)
//    * the 7-bit limb-split TC matrices: M1 / Minv are staged into shared
//      memory by stage_tc_matrices, the per-block Stage-2 matrices M2^(m) and
//      MA^(m) live in __device__ globals (wmma cannot read __constant__)
//    * ntt_stage2_wmma_batch<UB>       -- one 16x16x16 WMMA Stage-2 tile
//    * scalar_tail_fwd_dfs / _inv_dfs  -- DFS-ordered scalar Stage-2
//    * ntt_core / intt_core            -- the 16-staged-poly compute cores,
//                                         shared by all four variants
//    * ntt_regstage / intt_regstage    -- register <-> shared wrappers around
//                                         the cores, used by 2TC and TC-CC
//    * stage_tc_matrices / setup_tc_matrices (the latter probes the reference
//      layer code with unit vectors, so the matrices provably equal the real
//      linear map)
//
// ============================================================================
#ifndef NTT_KERNELS_CUH
#define NTT_KERNELS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

// ------------------------- ring / parameter set -----------------------------
// KYBER_K is overridable (-DKYBER_K=2/3/4).  The ring (n, q) is identical for
// every ML-KEM parameter set, so the kernels below are parameter-independent;
// K only selects the reference C namespace used for validation.
#ifndef KYBER_K
#define KYBER_K   4
#endif
#if KYBER_K < 2 || KYBER_K > 4
#error "KYBER_K must be 2, 3 or 4"
#endif
#define KYBER_N   256
#define KYBER_Q   3329
#define QINV      62209

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)
// ----------------------------- zetas --------------------------------------
static const int16_t zetas_host[128] = {
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
};
__constant__ int16_t d_zetas[128];

// ----------------------------- reductions ---------------------------------
__device__ __host__ __forceinline__ int16_t mont_reduce(int32_t a)
{
  int16_t t = (int16_t)((int16_t)a * (int16_t)QINV);
  t = (int16_t)((a - (int32_t)t*KYBER_Q) >> 16);
  return t;
}

__device__ __host__ __forceinline__ int16_t barrett(int16_t a)
{
  const int16_t v = ((1<<26) + KYBER_Q/2)/KYBER_Q;
  int16_t t = (int16_t)(((int32_t)v*a + (1<<25)) >> 26);
  t = (int16_t)(t*KYBER_Q);
  return (int16_t)(a - t);
}

// int32 Barrett reduction for the WMMA accumulator combine.  Same constants as
// barrett() but widened to the full int32 accumulator: S = sum_k A[k]*B[k] over
// a 16-term mma dot with A,B in [0,q), so |S| <= 16*(q-1)^2 ~ 1.77e8 and the
// reduction is exact.  Returns r == a (mod q) with r in (-q,q); callers
// normalise to [0,q) or centre as needed.
__device__ __host__ __forceinline__ int32_t barrett32(int32_t a)
{
  const int32_t v = ((1<<26) + KYBER_Q/2)/KYBER_Q;
  const int32_t t = (int32_t)(((int64_t)v*a + (1<<25)) >> 26);
  return (int32_t)(a - (int32_t)t*KYBER_Q);
}

#ifndef TC_UNSIGNED_BARRETT
#define TC_UNSIGNED_BARRETT 1
#endif
__device__ __host__ __forceinline__ int32_t barrett32u(int32_t a)
{
  const uint32_t m = (uint32_t)((1ull<<32) / KYBER_Q);   // floor(2^32/q)
#if defined(__CUDA_ARCH__)
  const uint32_t t = __umulhi((uint32_t)a, m);
#else
  const uint32_t t = (uint32_t)(((uint64_t)(uint32_t)a * m) >> 32);
#endif
  uint32_t r = (uint32_t)a - t * (uint32_t)KYBER_Q;
  if (r >= (uint32_t)KYBER_Q) {
    r -= (uint32_t)KYBER_Q;
  }
  return (int32_t)r;
}

// combine reduction: UB=1 -> unsigned Barrett (returns [0,q) directly);
// UB=0 -> signed barrett32 plus the usual `if (v<0) v += q` normalisation.
template<int UB>
__device__ __forceinline__ int32_t tc_combine(int32_t S)
{
  if (UB) {
    return barrett32u(S);
  }
  int32_t v = barrett32(S);
  if (v < 0) {
    v += KYBER_Q;
  }
  return v;
}

__device__ __host__ __forceinline__ int16_t fqmul(int16_t a, int16_t b)
{
  return mont_reduce((int32_t)a*b);
}


__constant__ int8_t  c_M1_lo[256],  c_M1_hi[256];
__constant__ int8_t  c_Minv_lo[256], c_Minv_hi[256];

// Per-block Stage-2 matrices M2^(m) (forward, len = 8,4,2) and their inverse
// mirror MA^(m) (iNTT, len = 2,4,8): 16 blocks x 16x16, lo/hi limbs each.
// __device__ globals rather than __constant__, because wmma::load_matrix_sync
// cannot read __constant__ memory.
__device__ int8_t  c_M2_lo[16*256],  c_M2_hi[16*256];
__device__ int8_t  c_MA_lo[16*256],  c_MA_hi[16*256];

// Default Stage-2 mode for the S2 template parameters: 1 = WMMA, 0 = scalar.
#ifndef TC_STAGE2_WMMA
#define TC_STAGE2_WMMA 1
#endif

#ifndef TC_SLOT_STRIDE
#define TC_SLOT_STRIDE (KYBER_N + 8)
#endif

// Fragment layout of the int8 mma.m16n8k16 (wmma 16x16x16): the (row,col) that
// each lane's 8 B / accumulator fragment elements own.
#define TC_FRAG_LAYOUT                                       \
  const int r0 = lane / 4, cg = lane % 4;                    \
  const int row_base = (lane % 4) * 4, col_base = lane / 4;  \
  const int b_cols[8] = {col_base, col_base, col_base, col_base,    \
                         col_base + 8, col_base + 8, col_base + 8,  \
                         col_base + 8};                             \
  const int b_rows[8] = {row_base, row_base + 1, row_base + 2, row_base + 3, \
                         row_base, row_base + 1, row_base + 2, row_base + 3}; \
  const int frag_row[8] = {r0, r0, r0 + 8, r0 + 8, r0, r0, r0 + 8, r0 + 8}; \
  const int frag_col[8] = {cg * 2, cg * 2 + 1, cg * 2, cg * 2 + 1,   \
                           cg * 2 + 8, cg * 2 + 9, cg * 2 + 8, cg * 2 + 9}

// ============================================================================
//  Stage-2 WMMA for one 16-coefficient block, applied to ALL 16 staged
//  polynomials at once (one 16x16x16 tile per block m).
// ============================================================================
template<int UB=0>
__device__ __forceinline__ void ntt_stage2_wmma_batch(int16_t* s_polys, int m,
                                                      const int8_t* Mlo, const int8_t* Mhi,
                                                      int8_t* sh_blo, int8_t* sh_bhi,
                                                      int32_t* sh_acc)
{
  (void)sh_blo;
  (void)sh_bhi;
  (void)sh_acc;

  const int lane = threadIdx.x & 31;
  TC_FRAG_LAYOUT;

  wmma::fragment<wmma::matrix_a,16,16,16,int8_t,wmma::row_major> Alo, Ahi;
  wmma::load_matrix_sync(Alo, Mlo, 16);
  wmma::load_matrix_sync(Ahi, Mhi, 16);

  // Inputs are provably in [0,q) (Stage-1 output forward, staged values
  // inverse), so the exact 7-bit split v = lo + 128*hi needs no %q here.
  wmma::fragment<wmma::matrix_b,16,16,16,int8_t,wmma::row_major> Blo, Bhi;
  wmma::fragment<wmma::accumulator,16,16,16,int32_t> C0, C1, C2;

  #pragma unroll
  for (int e = 0; e < 8; e++) {
    const int16_t val = s_polys[b_cols[e]*TC_SLOT_STRIDE + 16*m + b_rows[e]];
    Blo.x[e] = (int8_t)(val & 127);
    Bhi.x[e] = (int8_t)(val >> 7);
  }
  wmma::fill_fragment(C0, 0);
  wmma::fill_fragment(C1, 0);
  wmma::fill_fragment(C2, 0);
  wmma::mma_sync(C0, Alo, Blo, C0);        // weight 1
  wmma::mma_sync(C1, Alo, Bhi, C1);        // weight 128
  wmma::mma_sync(C1, Ahi, Blo, C1);        // weight 128, same accumulator
  wmma::mma_sync(C2, Ahi, Bhi, C2);        // weight 16384


  #pragma unroll
  for (int e = 0; e < 8; e++) {
    const int32_t S = C0.x[e] + 128*C1.x[e] + 16384*C2.x[e];
    s_polys[frag_col[e]*TC_SLOT_STRIDE + 16*m + frag_row[e]] = (int16_t)tc_combine<UB>(S);
  }
  __syncwarp();
}
// ============================================================================
//  DFS-ordered scalar tail (layers 5..7) on one 16-coefficient block: the 24
//  butterflies of the three-level CT tree are executed in depth-first
//  (preorder) order.  This is the scalar Stage-2 used when S2 = 0.
//
//  forward CT tail, preorder: len-8 root -> len-4 left -> len-2 (q0,q1) ->
//  len-4 right -> len-2 (q2,q3)
// ============================================================================
__device__ __forceinline__ void scalar_tail_fwd_dfs(int16_t b[16], int bb)
{
  { // len = 8 : root, pairs (j, j+8)
    const int16_t z = d_zetas[16 + bb];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int16_t t = fqmul(z, b[j+8]);
      b[j+8] = (int16_t)(b[j] - t);
      b[j]   = (int16_t)(b[j] + t);
    }
  }
  { // len = 4 : left half, pairs (j, j+4)
    const int16_t z = d_zetas[32 + 2*bb + 0];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int16_t t = fqmul(z, b[j+4]);
      b[j+4] = (int16_t)(b[j] - t);
      b[j]   = (int16_t)(b[j] + t);
    }
  }
  { // len = 2 : quarter q0 (points 0..3), pairs (0,2),(1,3)
    const int16_t z = d_zetas[64 + 4*bb + 0];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = fqmul(z, b[j+2]);
      b[j+2] = (int16_t)(b[j] - t);
      b[j]   = (int16_t)(b[j] + t);
    }
  }
  { // len = 2 : quarter q1 (points 4..7), pairs (4,6),(5,7)
    const int16_t z = d_zetas[64 + 4*bb + 1];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = fqmul(z, b[4+j+2]);
      b[4+j+2] = (int16_t)(b[4+j] - t);
      b[4+j]   = (int16_t)(b[4+j] + t);
    }
  }
  { // len = 4 : right half, pairs (8+j, 12+j)
    const int16_t z = d_zetas[32 + 2*bb + 1];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int16_t t = fqmul(z, b[8+j+4]);
      b[8+j+4] = (int16_t)(b[8+j] - t);
      b[8+j]   = (int16_t)(b[8+j] + t);
    }
  }
  { // len = 2 : quarter q2 (points 8..11), pairs (8,10),(9,11)
    const int16_t z = d_zetas[64 + 4*bb + 2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = fqmul(z, b[8+j+2]);
      b[8+j+2] = (int16_t)(b[8+j] - t);
      b[8+j]   = (int16_t)(b[8+j] + t);
    }
  }
  { // len = 2 : quarter q3 (points 12..15), pairs (12,14),(13,15)
    const int16_t z = d_zetas[64 + 4*bb + 3];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = fqmul(z, b[12+j+2]);
      b[12+j+2] = (int16_t)(b[12+j] - t);
      b[12+j]   = (int16_t)(b[12+j] + t);
    }
  }
}

// ============================================================================
//  Inverse GS tail (layers 5..7) on one 16-coefficient block: the mirrored
//  tree, again in preorder --
//  len-2 quarters (q0..q3) -> len-4 halves (left, right) -> len-8 root.
//  The sums are Barrett-reduced to stay in range.
// ============================================================================
__device__ __forceinline__ void scalar_tail_inv_dfs(int16_t b[16], int bb)
{
  { // len = 2 : quarter q0 (points 0..3)
    const int16_t z = d_zetas[127 - (4*bb + 0)];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = b[j];
      b[j]   = barrett((int16_t)(t + b[j+2]));
      b[j+2] = (int16_t)(b[j+2] - t);
      b[j+2] = fqmul(z, b[j+2]);
    }
  }
  { // len = 2 : quarter q1 (points 4..7)
    const int16_t z = d_zetas[127 - (4*bb + 1)];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = b[4+j];
      b[4+j]   = barrett((int16_t)(t + b[4+j+2]));
      b[4+j+2] = (int16_t)(b[4+j+2] - t);
      b[4+j+2] = fqmul(z, b[4+j+2]);
    }
  }
  { // len = 2 : quarter q2 (points 8..11)
    const int16_t z = d_zetas[127 - (4*bb + 2)];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = b[8+j];
      b[8+j]   = barrett((int16_t)(t + b[8+j+2]));
      b[8+j+2] = (int16_t)(b[8+j+2] - t);
      b[8+j+2] = fqmul(z, b[8+j+2]);
    }
  }
  { // len = 2 : quarter q3 (points 12..15)
    const int16_t z = d_zetas[127 - (4*bb + 3)];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      const int16_t t = b[12+j];
      b[12+j]   = barrett((int16_t)(t + b[12+j+2]));
      b[12+j+2] = (int16_t)(b[12+j+2] - t);
      b[12+j+2] = fqmul(z, b[12+j+2]);
    }
  }
  { // len = 4 : left half, pairs (j, j+4)
    const int16_t z = d_zetas[63 - (2*bb + 0)];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int16_t t = b[j];
      b[j]   = barrett((int16_t)(t + b[j+4]));
      b[j+4] = (int16_t)(b[j+4] - t);
      b[j+4] = fqmul(z, b[j+4]);
    }
  }
  { // len = 4 : right half, pairs (8+j, 12+j)
    const int16_t z = d_zetas[63 - (2*bb + 1)];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int16_t t = b[8+j];
      b[8+j]   = barrett((int16_t)(t + b[8+j+4]));
      b[8+j+4] = (int16_t)(b[8+j+4] - t);
      b[8+j+4] = fqmul(z, b[8+j+4]);
    }
  }
  { // len = 8 : root, pairs (j, j+8)
    const int16_t z = d_zetas[31 - bb];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int16_t t = b[j];
      b[j]   = barrett((int16_t)(t + b[j+8]));
      b[j+8] = (int16_t)(b[j+8] - t);
      b[j+8] = fqmul(z, b[j+8]);
    }
  }
}
// ============================================================================
//  Compute core: forward layers 1..4 (M1) + layers 5..7 (Stage-2), running on
//  one buffer of 16 staged polynomials that are TC_SLOT_STRIDE int16 apart.
//  sh_Mlo / sh_Mhi are the M1 lo/hi limb matrices staged by stage_tc_matrices.
//
//    BASIC = false : the staged values are full-range [0,q), so both limbs of
//                    the input tile are needed -> 4 mma per tile.
//    BASIC = true  : the staged values fit one signed int8 limb (packed-input
//                    forward, |v| <= 3), so only the constant needs the 7-bit
//                    split -> 2 mma per tile.
//    S2            : 1 = layers 5..7 as per-block WMMA tiles,
//                    0 = layers 5..7 as scalar butterflies.
//
//  B[r][c] = s_polys[p*TC_SLOT_STRIDE + r*16 + c] is polynomial p seen as a
//  16x16 row-major grid, built directly in registers and scattered straight
//  back, so there is no shared staging round trip and only one __syncwarp()
//  per polynomial.  All 32 lanes must call this in lockstep.
// ============================================================================
template<bool BASIC=false, int S2=TC_STAGE2_WMMA>
__device__ __forceinline__ void ntt_core(int16_t* s_polys,
                                         const int8_t* sh_Mlo, const int8_t* sh_Mhi)
{
  const int lane = threadIdx.x & 31;
  TC_FRAG_LAYOUT;

  wmma::fragment<wmma::matrix_a,16,16,16,int8_t,wmma::row_major> Alo, Ahi;
  wmma::load_matrix_sync(Alo, sh_Mlo, 16);
  wmma::load_matrix_sync(Ahi, sh_Mhi, 16);

  // ---- layers 1..4: M1 applied to each of the 16 staged polynomials -------
  for (int p = 0; p < 16; p++) {
    if (BASIC) {
      wmma::fragment<wmma::matrix_b,16,16,16,int8_t,wmma::row_major> Blo;
      wmma::fragment<wmma::accumulator,16,16,16,int32_t> C0, C1;

      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const int16_t val = s_polys[p*TC_SLOT_STRIDE + b_rows[e]*16 + b_cols[e]];
        Blo.x[e] = (int8_t)val;               // single signed limb
      }
      wmma::fill_fragment(C0, 0);
      wmma::fill_fragment(C1, 0);
      wmma::mma_sync(C0, Alo, Blo, C0);       // Alo * v  (weight 1)
      wmma::mma_sync(C1, Ahi, Blo, C1);       // Ahi * v  (weight 128)

      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const int32_t S = C0.x[e] + 128*C1.x[e];
        int32_t v = barrett32(S);
        if (v < 0) {
          v += KYBER_Q;
        }
        s_polys[p*TC_SLOT_STRIDE + frag_row[e]*16 + frag_col[e]] = (int16_t)v;
      }
    } else {
      wmma::fragment<wmma::matrix_b,16,16,16,int8_t,wmma::row_major> Blo, Bhi;
      wmma::fragment<wmma::accumulator,16,16,16,int32_t> C0, C1, C2;

      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const int16_t val = s_polys[p*TC_SLOT_STRIDE + b_rows[e]*16 + b_cols[e]];
        Blo.x[e] = (int8_t)(val & 127);
        Bhi.x[e] = (int8_t)(val >> 7);
      }
      wmma::fill_fragment(C0, 0);
      wmma::fill_fragment(C1, 0);
      wmma::fill_fragment(C2, 0);
      wmma::mma_sync(C0, Alo, Blo, C0);       // weight 1
      wmma::mma_sync(C1, Alo, Bhi, C1);       // weight 128
      wmma::mma_sync(C1, Ahi, Blo, C1);       // weight 128, same accumulator
      wmma::mma_sync(C2, Ahi, Bhi, C2);       // weight 16384

      #pragma unroll
      for (int e = 0; e < 8; e++) {
        const int32_t S = C0.x[e] + 128*C1.x[e] + 16384*C2.x[e];
        int32_t v = barrett32(S);
        if (v < 0) {
          v += KYBER_Q;
        }
        s_polys[p*TC_SLOT_STRIDE + frag_row[e]*16 + frag_col[e]] = (int16_t)v;
      }
    }
    __syncwarp();
  }

  // ---- layers 5..7: Stage-2 ---------------------------------------------
  if (S2) {
    // one 16x16x16 tile per block m applies M2^(m) to all 16 staged polys
    for (int m = 0; m < 16; m++) {
      ntt_stage2_wmma_batch<TC_UNSIGNED_BARRETT>(
          s_polys, m,
          c_M2_lo + (size_t)m*256, c_M2_hi + (size_t)m*256,
          (int8_t*)s_polys, (int8_t*)s_polys, (int32_t*)s_polys);
    }
  } else {
    // scalar butterflies; 32 lanes cover 2 polynomials x 16 blocks per step
    for (int pp = 0; pp < 16; pp += 2) {
      const int my_poly = pp + (lane >> 4);
      const int bb = lane & 15;
      int16_t* rp = s_polys + my_poly*TC_SLOT_STRIDE + bb*16;
      int16_t r_local[16];

      #pragma unroll
      for (int j = 0; j < 16; j++) {
        r_local[j] = rp[j];
      }
      scalar_tail_fwd_dfs(r_local, bb);
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        rp[j] = r_local[j];
      }
    }
    __syncwarp();
  }
}
// ============================================================================
//  ntt_regstage: register <-> shared staging / read-back around ntt_core, for
//  call sites that already hold their polynomial in registers.  One warp owns
//  32 tasks with ONE polynomial each, so the 32 lanes stage the 16 polynomials
//  of a round out of their own registers and read their own result back.  This
//  is the forward used by the 2TC and TC-CC kernels.  s_blo / s_bhi / s_acc are
//  unused and only exist so all wrappers share one call signature.
// ============================================================================
template<bool BASIC=false, int S2=TC_STAGE2_WMMA>
__device__ __forceinline__ void ntt_regstage(int16_t (*poly)[KYBER_N], int npoly,
                                             const int8_t* sh_Mlo, const int8_t* sh_Mhi,
                                             int16_t* s_polys, int8_t* s_blo,
                                             int8_t* s_bhi, int32_t* s_acc)
{
  const int lane = threadIdx.x & 31;
  (void)s_blo;
  (void)s_bhi;
  (void)s_acc;

  for (int i = 0; i < npoly; i++) {
    for (int round = 0; round < 2; round++) {
      const int slot = lane - round*16;

      if ((unsigned)slot < 16u) {
        // Stage this lane's polynomial.  The inputs are < q, so the 7-bit split
        // is exact (|v| < 16384) and no %q normalisation is needed.
        const int4* src = (const int4*)poly[i];
        int4*       dst = (int4*)(s_polys + slot*TC_SLOT_STRIDE);
        #pragma unroll
        for (int w4 = 0; w4 < KYBER_N/8; w4++) {
          dst[w4] = src[w4];
        }
      }
      __syncwarp();

      ntt_core<BASIC, S2>(s_polys, sh_Mlo, sh_Mhi);

      if ((unsigned)slot < 16u) {
        const int4* src = (const int4*)(s_polys + slot*TC_SLOT_STRIDE);
        int4*       dst = (int4*)poly[i];
        #pragma unroll
        for (int w4 = 0; w4 < KYBER_N/8; w4++) {
          dst[w4] = src[w4];
        }
      }
      __syncwarp();
    }
  }
}

// ============================================================================
//  Compute core (inverse): layers 5..7 (Stage-A) + layers 1..4 (Minv, which
//  also applies the final 1441 scaling), running on one buffer of 16 staged
//  polynomials that are TC_SLOT_STRIDE int16 apart.  sh_Mlo / sh_Mhi are the
//  Minv lo/hi limb matrices staged by stage_tc_matrices.
//
//    S2 : 1 = Stage-A as per-block WMMA tiles, 0 = scalar GS butterflies.
// ============================================================================
template<int S2=TC_STAGE2_WMMA>
__device__ __forceinline__ void intt_core(int16_t* s_polys,
                                          const int8_t* sh_Mlo, const int8_t* sh_Mhi)
{
  const int lane = threadIdx.x & 31;
  TC_FRAG_LAYOUT;

  wmma::fragment<wmma::matrix_a,16,16,16,int8_t,wmma::row_major> Alo, Ahi;
  wmma::load_matrix_sync(Alo, sh_Mlo, 16);
  wmma::load_matrix_sync(Ahi, sh_Mhi, 16);

  // ---- Stage-A: layers 5..7 (the bottom three layers) ---------------------
  if (S2) {
    // one 16x16x16 tile per block m applies MA^(m) to all 16 staged polys
    for (int m = 0; m < 16; m++) {
      ntt_stage2_wmma_batch<0>(
          s_polys, m,
          c_MA_lo + (size_t)m*256, c_MA_hi + (size_t)m*256,
          (int8_t*)s_polys, (int8_t*)s_polys, (int32_t*)s_polys);
    }
  } else {
    // scalar GS butterflies (len = 2,4,8); results normalised to [0,q) for Minv
    for (int pp = 0; pp < 16; pp += 2) {
      const int my_poly = pp + (lane >> 4);
      const int bb = lane & 15;
      int16_t* rp = s_polys + my_poly*TC_SLOT_STRIDE + bb*16;
      int16_t r_local[16];

      #pragma unroll
      for (int j = 0; j < 16; j++) {
        r_local[j] = rp[j];
      }
      scalar_tail_inv_dfs(r_local, bb);
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        int16_t v = r_local[j];
        if (v < 0) {
          v += KYBER_Q;
        }
        rp[j] = v;
      }
    }
    __syncwarp();
  }

  // ---- layers 1..4: Minv applied to each of the 16 staged polynomials ----
  // The inputs are in (-q,q), so the plain 7-bit split is exact and needs no
  // %q (|val| < q < q*128 keeps the high limb inside int8 range).
  for (int p = 0; p < 16; p++) {
    wmma::fragment<wmma::matrix_b,16,16,16,int8_t,wmma::row_major> Blo, Bhi;
    wmma::fragment<wmma::accumulator,16,16,16,int32_t> C0, C1, C2;

    #pragma unroll
    for (int e = 0; e < 8; e++) {
      const int16_t val = s_polys[p*TC_SLOT_STRIDE + b_rows[e]*16 + b_cols[e]];
      Blo.x[e] = (int8_t)(val & 127);
      Bhi.x[e] = (int8_t)(val >> 7);
    }
    wmma::fill_fragment(C0, 0);
    wmma::fill_fragment(C1, 0);
    wmma::fill_fragment(C2, 0);
    wmma::mma_sync(C0, Alo, Blo, C0);       // weight 1
    wmma::mma_sync(C1, Alo, Bhi, C1);       // weight 128
    wmma::mma_sync(C1, Ahi, Blo, C1);       // weight 128, same accumulator
    wmma::mma_sync(C2, Ahi, Bhi, C2);       // weight 16384

    #pragma unroll
    for (int e = 0; e < 8; e++) {
      const int32_t S = C0.x[e] + 128*C1.x[e] + 16384*C2.x[e];
      int32_t v = tc_combine<TC_UNSIGNED_BARRETT>(S);
      if (v > KYBER_Q/2) {
        v -= KYBER_Q;                       // centre, matches the reference
      }
      s_polys[p*TC_SLOT_STRIDE + frag_row[e]*16 + frag_col[e]] = (int16_t)v;
    }
    __syncwarp();
  }
}

template<int S2=TC_STAGE2_WMMA>
__device__ __forceinline__ void intt_regstage(int16_t (*poly)[KYBER_N], int npoly,
                                              const int8_t* sh_Mlo, const int8_t* sh_Mhi,
                                              int16_t* s_polys, int8_t* s_blo,
                                              int8_t* s_bhi, int32_t* s_acc)
{
  const int lane = threadIdx.x & 31;
  (void)s_blo;
  (void)s_bhi;
  (void)s_acc;

  for (int i = 0; i < npoly; i++) {
    for (int round = 0; round < 2; round++) {
      const int slot = lane - round*16;

      if ((unsigned)slot < 16u) {
        // Stage this lane's polynomial (|v| < q, so the 7-bit split is exact)
        const int4* src = (const int4*)poly[i];
        int4*       dst = (int4*)(s_polys + slot*TC_SLOT_STRIDE);
        #pragma unroll
        for (int w4 = 0; w4 < KYBER_N/8; w4++) {
          dst[w4] = src[w4];
        }
      }
      __syncwarp();

      intt_core<S2>(s_polys, sh_Mlo, sh_Mhi);

      if ((unsigned)slot < 16u) {
        const int4* src = (const int4*)(s_polys + slot*TC_SLOT_STRIDE);
        int4*       dst = (int4*)poly[i];
        #pragma unroll
        for (int w4 = 0; w4 < KYBER_N/8; w4++) {
          dst[w4] = src[w4];
        }
      }
      __syncwarp();
    }
  }
}


// ============================================================================
//  Shared staging of the four constant matrices.  The dynamic shared block
//  starts with [M1_lo | M1_hi | Minv_lo | Minv_hi] (4 x 256 int8 =
//  TC_MAT_BYTES bytes); the batch kernels place their per-warp polynomial
//  buffer after it.  The copy exists because wmma::load_matrix_sync cannot read
//  __constant__ memory.
// ============================================================================
#define TC_MAT_BYTES (4*256)

__device__ __forceinline__ void stage_tc_matrices(uint8_t* smem)
{
  int8_t* M = (int8_t*)smem;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    M[      i] = c_M1_lo[i];
    M[256 + i] = c_M1_hi[i];
    M[512 + i] = c_Minv_lo[i];
    M[768 + i] = c_Minv_hi[i];
  }
  __syncthreads();
}
// ============================================================================
//  HOST side: build the TC matrices by probing the reference layer code with
//  unit vectors, so each matrix provably equals the real linear map.
// ============================================================================
static int16_t h_mont(int32_t a)
{
  return mont_reduce(a);
}

static int16_t h_fqmul(int16_t a, int16_t b)
{
  return h_mont((int32_t)a*b);
}

// NTT layers 1..4 only (len = 128,64,32,16), k = 1..15
static void h_ntt_L1_4(int16_t r[256])
{
  unsigned k = 1;
  for (unsigned len = 128; len >= 16; len >>= 1) {
    for (unsigned start = 0; start < 256; start += (len << 1)) {
      const int16_t z = zetas_host[k++];
      for (unsigned j = start; j < start + len; j++) {
        const int16_t t = h_fqmul(z, r[j+len]);
        r[j+len] = (int16_t)(r[j] - t);
        r[j]     = (int16_t)(r[j] + t);
      }
    }
  }
}

// NTT layers 5..7 (len = 8,4,2), k = 16..127: per-16-block twiddles, i.e.
// exactly the scalar Stage-2 of the device kernel (16 independent blocks).
static void h_ntt_L5_7(int16_t r[256])
{
  for (int b = 0; b < 16; b++) {
    int16_t* blk = r + 16*b;

    { // len = 8
      const int16_t z = zetas_host[16 + b];
      for (int j = 0; j < 8; j++) {
        const int16_t t = h_fqmul(z, blk[j+8]);
        blk[j+8] = (int16_t)(blk[j] - t);
        blk[j]   = (int16_t)(blk[j] + t);
      }
    }
    for (int s = 0; s < 2; s++) {           // len = 4
      const int16_t z = zetas_host[32 + 2*b + s];
      int16_t* p = blk + 8*s;
      for (int j = 0; j < 4; j++) {
        const int16_t t = h_fqmul(z, p[j+4]);
        p[j+4] = (int16_t)(p[j] - t);
        p[j]   = (int16_t)(p[j] + t);
      }
    }
    for (int s = 0; s < 4; s++) {           // len = 2
      const int16_t z = zetas_host[64 + 4*b + s];
      int16_t* p = blk + 4*s;
      for (int j = 0; j < 2; j++) {
        const int16_t t = h_fqmul(z, p[j+2]);
        p[j+2] = (int16_t)(p[j] - t);
        p[j]   = (int16_t)(p[j] + t);
      }
    }
  }
}
// iNTT layers 5..7 (len = 2,4,8, GS), k = 127..16: per-block, i.e. exactly the
// device scalar Stage-A (including the Barrett reduction).
static void h_gs_L2_8(int16_t r[256])
{
  const int16_t v = ((1<<26) + KYBER_Q/2)/KYBER_Q;

  for (int b = 0; b < 16; b++) {
    int16_t* blk = r + 16*b;

    for (int s = 0; s < 4; s++) {           // len = 2
      const int16_t z = zetas_host[127 - (4*b + (3-s))];
      int16_t* p = blk + 4*(3-s);
      for (int j = 0; j < 2; j++) {
        const int16_t t = p[j];
        const int16_t bt = (int16_t)(((int32_t)v*(t + p[j+2]) + (1<<25)) >> 26);
        p[j]   = (int16_t)((t + p[j+2]) - (int16_t)(bt*KYBER_Q));
        p[j+2] = (int16_t)(p[j+2] - t);
        p[j+2] = h_fqmul(z, p[j+2]);
      }
    }
    for (int s = 0; s < 2; s++) {           // len = 4
      const int16_t z = zetas_host[63 - (2*b + (1-s))];
      int16_t* p = blk + 8*(1-s);
      for (int j = 0; j < 4; j++) {
        const int16_t t = p[j];
        const int16_t bt = (int16_t)(((int32_t)v*(t + p[j+4]) + (1<<25)) >> 26);
        p[j]   = (int16_t)((t + p[j+4]) - (int16_t)(bt*KYBER_Q));
        p[j+4] = (int16_t)(p[j+4] - t);
        p[j+4] = h_fqmul(z, p[j+4]);
      }
    }
    {                                       // len = 8
      const int16_t z = zetas_host[31 - b];
      for (int j = 0; j < 8; j++) {
        const int16_t t = blk[j];
        const int16_t bt = (int16_t)(((int32_t)v*(t + blk[j+8]) + (1<<25)) >> 26);
        blk[j]   = (int16_t)((t + blk[j+8]) - (int16_t)(bt*KYBER_Q));
        blk[j+8] = (int16_t)(blk[j+8] - t);
        blk[j+8] = h_fqmul(z, blk[j+8]);
      }
    }
  }
}

// iNTT layers 1..4 (len = 16..128) plus the final 1441 scaling, k = 15..1
static void h_invntt_L4_7(int16_t r[256])
{
  const int16_t v = ((1<<26) + KYBER_Q/2)/KYBER_Q;
  int k = 15;

  for (unsigned len = 16; len <= 128; len <<= 1) {
    for (unsigned start = 0; start < 256; start += (len << 1)) {
      const int16_t z = zetas_host[k--];
      for (unsigned j = start; j < start + len; j++) {
        const int16_t t = r[j];
        const int16_t s = (int16_t)(t + r[j+len]);
        // Barrett to stay in range, matching the device path
        const int16_t bt = (int16_t)(((int32_t)v*s + (1<<25)) >> 26);
        r[j]     = (int16_t)(s - (int16_t)(bt*KYBER_Q));
        r[j+len] = (int16_t)(r[j+len] - t);
        r[j+len] = h_fqmul(z, r[j+len]);
      }
    }
  }
  for (unsigned j = 0; j < 256; j++) {
    r[j] = h_fqmul(r[j], 1441);
  }
}

// 7-bit limb split used by the table build and the device B-fills:
// lo = v&127, hi = v>>7 (lo in [0,127], hi in [0,26]); exact, v = lo + 128*hi.
static inline void tc_split(int16_t v, int8_t& lo, int8_t& hi)
{
  lo = (int8_t)(v & 127);
  hi = (int8_t)(v >> 7);
}

// ============================================================================
//  Build all TC matrices on the host by pushing unit vectors through the
//  reference layer code, then upload them.
//
//  M1 / Minv: a polynomial is viewed as IN[m][c] = r[c + 16m] and layers 1..4
//  act only on the m index, so probing with a unit vector e_m in column c = 0
//  yields the whole map.
//
//  M2^(m) / MA^(m): within a 16-coefficient block those layers never leave the
//  block, so probing a unit vector at 16m+j and reading 16m+c yields M[m][c][j].
// ============================================================================
static void setup_tc_matrices()
{
  int16_t M1[256], Minv[256];

  for (int m = 0; m < 16; m++) {
    int16_t vec[256];
    memset(vec, 0, sizeof(vec));
    vec[16*m] = 1;                          // unit vector e_m in the m index
    h_ntt_L1_4(vec);
    for (int row = 0; row < 16; row++) {
      int16_t v = (int16_t)(vec[16*row] % KYBER_Q);
      if (v < 0) {
        v += KYBER_Q;
      }
      M1[row*16 + m] = v;                   // OUT[row] = sum_m M1[row][m]*IN[m]
    }
  }
  for (int m = 0; m < 16; m++) {
    int16_t vec[256];
    memset(vec, 0, sizeof(vec));
    vec[16*m] = 1;
    h_invntt_L4_7(vec);
    for (int row = 0; row < 16; row++) {
      int16_t v = (int16_t)(vec[16*row] % KYBER_Q);
      if (v < 0) {
        v += KYBER_Q;
      }
      Minv[row*16 + m] = v;
    }
  }

  // split each entry into a 7-bit lo limb and a 128*hi limb; both fit int8
  // because q-1 = 3328
  int8_t M1lo[256], M1hi[256], Milo[256], Mihi[256];
  for (int i = 0; i < 256; i++) {
    tc_split(M1[i],   M1lo[i], M1hi[i]);
    tc_split(Minv[i], Milo[i], Mihi[i]);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_M1_lo,   M1lo, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_M1_hi,   M1hi, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_Minv_lo, Milo, 256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_Minv_hi, Mihi, 256));

  // ---- per-block Stage-2 matrices M2^(m) (forward len = 8,4,2) and
  //      MA^(m) (inverse GS len = 2,4,8) ----
  int8_t M2lo[16*256], M2hi[16*256], MAlo[16*256], MAhi[16*256];

  for (int m = 0; m < 16; m++) {
    for (int j = 0; j < 16; j++) {
      { // forward
        int16_t vec[256];
        memset(vec, 0, sizeof(vec));
        vec[16*m + j] = 1;
        h_ntt_L5_7(vec);
        for (int c = 0; c < 16; c++) {
          int16_t v = (int16_t)(vec[16*m + c] % KYBER_Q);
          if (v < 0) {
            v += KYBER_Q;
          }
          const int idx = m*256 + c*16 + j;
          tc_split(v, M2lo[idx], M2hi[idx]);
        }
      }
      { // inverse
        int16_t vec[256];
        memset(vec, 0, sizeof(vec));
        vec[16*m + j] = 1;
        h_gs_L2_8(vec);
        for (int c = 0; c < 16; c++) {
          int16_t v = (int16_t)(vec[16*m + c] % KYBER_Q);
          if (v < 0) {
            v += KYBER_Q;
          }
          const int idx = m*256 + c*16 + j;
          tc_split(v, MAlo[idx], MAhi[idx]);
        }
      }
    }
  }
  CUDA_CHECK(cudaMemcpyToSymbol(c_M2_lo, M2lo, 16*256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_M2_hi, M2hi, 16*256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_MA_lo, MAlo, 16*256));
  CUDA_CHECK(cudaMemcpyToSymbol(c_MA_hi, MAhi, 16*256));
}

#endif // NTT_KERNELS_CUH
