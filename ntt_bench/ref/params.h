#ifndef PARAMS_H
#define PARAMS_H

#ifndef KYBER_K
#define KYBER_K 3	/* Change this for different security strengths */
#endif


/* Don't change parameters below this line */
#if   (KYBER_K == 2)
#define KYBER_NAMESPACE(s) pqcrystals_kyber512_ref_##s
#elif (KYBER_K == 3)
#define KYBER_NAMESPACE(s) pqcrystals_kyber768_ref_##s
#elif (KYBER_K == 4)
#define KYBER_NAMESPACE(s) pqcrystals_kyber1024_ref_##s
#else
#error "KYBER_K must be in {2,3,4}"
#endif

#define KYBER_N 256
#define KYBER_Q 3329

/* KEM-only parameters (compression sizes, ETA1/ETA2, key/ciphertext sizes)
   are not needed by the standalone NTT/iNTT benchmark and have been removed. */

#endif
