#include <x86intrin.h>

#include "add.h"

#if defined(__AVX__)
__m256 _value_add256(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
}
__m256 (*value_add256)(__m256, __m256) = &_value_add256;
__m256 _inptr_add256(const float* a, const float* b) {
  return _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
}
__m256 (*inptr_add256)(const float*, const float*) = &_inptr_add256;
void _outptr_add256(__m256 a, __m256 b, float* dst) {
  _mm256_storeu_ps(dst, _mm256_add_ps(a, b));
}
void (*outptr_add256)(__m256, __m256, float*) = &_outptr_add256;
#else
void (*value_add256)() = NULL;
void (*inptr_add256)() = NULL;
void (*outptr_add256)() = NULL;
#endif

void _ptr_add256(const float* a, const float* b, float* dst) {
#if defined(__AVX__)
  _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
#elif defined(__SSE2__)
  for (int i = 0; i < 8; i += 4) {
    _mm_storeu_ps(dst + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
  }
#else
  for (int i = 0; i < 8; ++i) {
    dst[i] = a[i] + b[i];
  }
#endif
}
void (*ptr_add256)(const float*, const float*, float*) = &_ptr_add256;
