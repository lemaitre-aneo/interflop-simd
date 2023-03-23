#include <x86intrin.h>

#include "add.h"

#if defined(__SSE2__)
__m128 _value_add128(__m128 a, __m128 b) {
  return _mm_add_ps(a, b);
}
__m128 (*value_add128)(__m128, __m128) = &_value_add128;
__m128 _inptr_add128(const float* a, const float* b) {
  return _mm_add_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
}
__m128 (*inptr_add128)(const float*, const float*) = &_inptr_add128;
void _outptr_add128(__m128 a, __m128 b, float* dst) {
  _mm_storeu_ps(dst, _mm_add_ps(a, b));
}
void (*outptr_add128)(__m128, __m128, float*) = &_outptr_add128;
#else
// Set those symbols to NULL to signal the lib user that they are not implemented
void (*value_add128)() = NULL;
void (*inptr_add128)() = NULL;
void (*outptr_add128)() = NULL;
#endif

void _ptr_add128(const float* a, const float* b, float* dst) {
#if defined(__SSE2__)
  _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
#else
  for (int i = 0; i < 4; ++i) {
    dst[i] = a[i] + b[i];
  }
#endif
}
void (*ptr_add128)(const float*, const float*, float*) = &_ptr_add128;
