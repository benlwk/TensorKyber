# ============================================================================
#  Standalone NTT / iNTT benchmark -- Kyber-512 (N=256, q=3329)
#
#  Only three variants are built and timed:
#    * TC-v7 packed tern (scalar-S2)   -- 2-bit packed ternary input, BASIC M1
#    * TC-v4 cp.async WMMA-S2          -- cp.async single-buffer, WMMA Stage-2
#    * TC-v4 cp.async scalar-S2        -- cp.async single-buffer, scalar Stage-2
#
#  The baseline (BFS ref), EDFS-1-6 and TC-v3 batch32 paths of the original
#  bench_ntt.cu are intentionally NOT present here.
#
#  Defaults: Kyber-512 (K=2), batch = 262144 polynomials, 30 timed runs,
#            mean AND median reported for every variant.
#
#     make                 build ./ntt_bench
#     make run             run with the defaults (262144 polys, 30 runs)
#     make K=3 run         Kyber-768 instead of Kyber-512
#     make NPOLY=65536 RUNS=10 run
# ============================================================================
ARCH ?= sm_89
K    ?= 2
NVCC  = nvcc
CC    = gcc
CFLAGS  = -O3 -DKYBER_K=$(K)
NVFLAGS = -O3 -arch=$(ARCH) -std=c++14 -DKYBER_K=$(K)

REF    = ref
REFSRC = $(REF)/ntt.c $(REF)/reduce.c
REFOBJ = $(patsubst $(REF)/%.c,%.o,$(REFSRC))

# benchmark parameters (overridable on the command line)
NPOLY ?= 262144
RUNS  ?= 30

all: ntt_bench

# ---- reference NTT/iNTT objects (namespaced by -DKYBER_K) -------------------
$(REFOBJ): %.o: $(REF)/%.c $(REF)/params.h
	$(CC) $(CFLAGS) -c $< -o $@

ntt_bench: ntt_bench.cu ntt_kernels.cuh $(REFOBJ)
	$(NVCC) $(NVFLAGS) ntt_bench.cu $(REFOBJ) -o $@

run: ntt_bench
	./ntt_bench $(NPOLY) $(RUNS)

clean:
	rm -f *.o ntt_bench

.PHONY: all run clean
