#include <x86intrin.h>

#define VARIANT_REF 0
#define VARIANT_VALUE 1
#define VARIANT_INPTR 2
#define VARIANT_OUTPTR 3
#define VARIANT_PTR 4

#if VARIANT == VARIANT_VALUE
extern float (*value_add)(float, float);
#endif
#if VARIANT == VARIANT_INPTR
extern float (*inptr_add)(const float*, const float*);
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add)(float, float, float*);
#endif
#if VARIANT == VARIANT_PTR
extern void (*ptr_add)(const float*, const float*, float*);
#endif

#if defined(__SSE__)
#if VARIANT == VARIANT_VALUE
extern __m128 (*value_add128)(__m128, __m128);
#endif
#if VARIANT == VARIANT_INPTR
extern __m128 (*inptr_add128)(const float*, const float*);
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add128)(__m128, __m128, float*);
#endif
#else
#if VARIANT == VARIANT_VALUE
extern void (*value_add128)();
#endif
#if VARIANT == VARIANT_INPTR
extern void (*inptr_add128)();
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add128)();
#endif
#endif
#if VARIANT == VARIANT_PTR
extern void (*ptr_add128)(const float*, const float*, float*);
#endif

#if defined(__AVX__)
#if VARIANT == VARIANT_VALUE
extern __m256 (*value_add256)(__m256, __m256);
#endif
#if VARIANT == VARIANT_INPTR
extern __m256 (*inptr_add256)(const float*, const float*);
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add256)(__m256, __m256, float*);
#endif
#else
#if VARIANT == VARIANT_VALUE
extern void (*value_add256)();
#endif
#if VARIANT == VARIANT_INPTR
extern void (*inptr_add256)();
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add256)();
#endif
#endif
#if VARIANT == VARIANT_PTR
extern void (*ptr_add256)(const float*, const float*, float*);
#endif

#if defined(__AVX512F__)
#if VARIANT == VARIANT_VALUE
extern __m512 (*value_add512)(__m512, __m512);
#endif
#if VARIANT == VARIANT_INPTR
extern __m512 (*inptr_add512)(const float*, const float*);
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add512)(__m512, __m512, float*);
#endif
#else
#if VARIANT == VARIANT_VALUE
extern void (*value_add512)();
#endif
#if VARIANT == VARIANT_INPTR
extern void (*inptr_add512)();
#endif
#if VARIANT == VARIANT_OUTPTR
extern void (*outptr_add512)();
#endif
#endif
#if VARIANT == VARIANT_PTR
extern void (*ptr_add512)(const float*, const float*, float*);
#endif
