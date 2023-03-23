#include <x86intrin.h>

#include "add.h"

#if defined(__AVX512F__)
#if VARIANT == VARIANT_VALUE
__m512 _value_add512(__m512 a, __m512 b) {
  return _mm512_add_ps(a, b);
}
__m512 (*value_add512)(__m512, __m512) = &_value_add512;
#endif
#if VARIANT == VARIANT_INPTR
__m512 _inptr_add512(const float* a, const float* b) {
  return _mm512_add_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
}
__m512 (*inptr_add512)(const float*, const float*) = &_inptr_add512;
#endif
#if VARIANT == VARIANT_OUTPTR
void _outptr_add512(__m512 a, __m512 b, float* dst) {
  _mm512_storeu_ps(dst, _mm512_add_ps(a, b));
}
void (*outptr_add512)(__m512, __m512, float*) = &_outptr_add512;
#endif
#else
void (*value_add512)() = NULL;
void (*inptr_add512)() = NULL;
void (*outptr_add512)() = NULL;
#endif

#if VARIANT == VARIANT_PTR
void _ptr_add512(const float* a, const float* b, float* dst) {
#if defined(__AV512F__)
  _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b)));
#elif defined(__AVX__)
  for (int i = 0; i < 16; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
#elif defined(__SSE2__)
  for (int i = 0; i < 16; i += 4) {
    _mm_storeu_ps(dst + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
  }
#else
  for (int i = 0; i < 16; ++i) {
    dst[i] = a[i] + b[i];
  }
#endif
}
void (*ptr_add512)(const float*, const float*, float*) = &_ptr_add512;
#endif
