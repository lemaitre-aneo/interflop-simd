#include <x86intrin.h>

extern float (*value_add)(float, float);
extern float (*inptr_add)(const float*, const float*);
extern void (*outptr_add)(float, float, float*);
extern void (*ptr_add)(const float*, const float*, float*);

#if defined(__SSE2__)
extern __m128 (*value_add128)(__m128, __m128);
extern __m128 (*inptr_add128)(const float*, const float*);
extern void (*outptr_add128)(__m128, __m128, float*);
#endif
extern void (*ptr_add128)(const float*, const float*, float*);

#if defined(__AVX__)
extern __m256 (*value_add256)(__m256, __m256);
extern __m256 (*inptr_add256)(const float*, const float*);
extern void (*outptr_add256)(__m256, __m256, float*);
#endif
extern void (*ptr_add256)(const float*, const float*, float*);

#if defined(__AVX512F__)
extern __m512 (*value_add512)(__m512, __m512);
extern __m512 (*inptr_add512)(const float*, const float*);
extern void (*outptr_add512)(__m512, __m512, float*);
#endif
extern void (*ptr_add512)(const float*, const float*, float*);
