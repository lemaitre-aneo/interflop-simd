#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#include <x86intrin.h>
#include "add.h"

#if DEBUG
#define VERBOSE(...) (__VA_ARGS__)
#else
#define VERBOSE(...) ((void)0)
#endif

#if FORCE_LOAD
#define OPTIMIZATION_BARRIER(T, x) ({ \
      T v = x;                        \
      asm ("":"+v"(v));               \
      v;                              \
    })
#else
#define OPTIMIZATION_BARRIER(T, x) (x)
#endif


#if VARIANT == VARIANT_REF
void ref_combine_vec(float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ref scalar version\n"));
  for (int i = 0; i < n; ++i) {
    float a, b, c;
    a = OPTIMIZATION_BARRIER(float, A[i]);
    b = OPTIMIZATION_BARRIER(float, B[i]);
    c = a + b;
    A[i] = OPTIMIZATION_BARRIER(float, c);
  }
}
#endif
#if VARIANT == VARIANT_VALUE
void value_combine_vec(float(*f)(float, float), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use value scalar version\n"));
  for (int i = 0; i < n; ++i) {
    float a, b, c;
    a = OPTIMIZATION_BARRIER(float, A[i]);
    b = OPTIMIZATION_BARRIER(float, B[i]);
    c = f(a, b);
    A[i] = OPTIMIZATION_BARRIER(float, c);
  }
}
#endif
#if VARIANT == VARIANT_INPTR
void inptr_combine_vec(float(*f)(const float*, const float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use inptr scalar version\n"));
  for (int i = 0; i < n; ++i) {
    float a, b, c;
    a = OPTIMIZATION_BARRIER(float, A[i]);
    b = OPTIMIZATION_BARRIER(float, B[i]);
    c = f(&a, &b);
    A[i] = OPTIMIZATION_BARRIER(float, c);
  }
}
#endif
#if VARIANT == VARIANT_OUTPTR
void outptr_combine_vec(void(*f)(float, float, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use outptr scalar version\n"));
  for (int i = 0; i < n; ++i) {
    float a, b, c;
    a = OPTIMIZATION_BARRIER(float, A[i]);
    b = OPTIMIZATION_BARRIER(float, B[i]);
    f(a, b, &c);
    A[i] = OPTIMIZATION_BARRIER(float, c);
  }
}
#endif
#if VARIANT == VARIANT_PTR
void ptr_combine_vec(void(*f)(const float*, const float*, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ptr scalar version\n"));
  for (int i = 0; i < n; ++i) {
    float a, b, c;
    a = OPTIMIZATION_BARRIER(float, A[i]);
    b = OPTIMIZATION_BARRIER(float, B[i]);
    f(&a, &b, &c);
    A[i] = OPTIMIZATION_BARRIER(float, c);
  }
}
#endif

#if defined(__SSE2__)
#if VARIANT == VARIANT_REF
void ref_combine_vec128(float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ref VEC128 version\n"));
  for (int i = 0; i < n; i += 4) {
    __m128 a, b, c;
    a = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&B[i]));
    c = _mm_add_ps(a, b);
    _mm_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m128, c));
  }
}
#endif
#if VARIANT == VARIANT_VALUE
void value_combine_vec128(__m128(*f)(__m128, __m128), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use value VEC128 version\n"));
  for (int i = 0; i < n; i += 4) {
    __m128 a, b, c;
    a = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&B[i]));
    c = f(a, b);
    _mm_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m128, c));
  }
}
#endif
#if VARIANT == VARIANT_INPTR
void inptr_combine_vec128(__m128(*f)(const float*, const float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use inptr VEC128 version\n"));
  for (int i = 0; i < n; i += 4) {
    __m128 a, b, c;
    a = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&B[i]));
    c = f((float*)&a, (float*)&b);
    _mm_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m128, c));
  }
}
#endif
#if VARIANT == VARIANT_OUTPTR
void outptr_combine_vec128(void(*f)(__m128, __m128, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use outptr VEC128 version\n"));
  for (int i = 0; i < n; i += 4) {
    __m128 a, b, c;
    a = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&B[i]));
    f(a, b, (float*)&c);
    _mm_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m128, c));
  }
}
#endif
#if VARIANT == VARIANT_PTR
void ptr_combine_vec128(void(*f)(const float*, const float*, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ptr VEC128 version\n"));
  for (int i = 0; i < n; i += 4) {
    __m128 a, b, c;
    a = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m128, _mm_loadu_ps(&B[i]));
    f((float*)&a, (float*)&b, (float*)&c);
    _mm_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m128, c));
  }
}
#endif
#endif

#if defined(__AVX__)
#if VARIANT == VARIANT_REF
void ref_combine_vec256(float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ref VEC256 version\n"));
  for (int i = 0; i < n; i += 8) {
    __m256 a, b, c;
    a = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&B[i]));
    c = _mm256_add_ps(a, b);
    _mm256_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m256, c));
  }
}
#endif
#if VARIANT == VARIANT_VALUE
void value_combine_vec256(__m256(*f)(__m256, __m256), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use value VEC256 version\n"));
  for (int i = 0; i < n; i += 8) {
    __m256 a, b, c;
    a = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&B[i]));
    c = f(a, b);
    _mm256_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m256, c));
  }
}
#endif
#if VARIANT == VARIANT_INPTR
void inptr_combine_vec256(__m256(*f)(const float*, const float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use inptr VEC256 version\n"));
  for (int i = 0; i < n; i += 8) {
    __m256 a, b, c;
    a = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&B[i]));
    c = f((float*)&a, (float*)&b);
    _mm256_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m256, c));
  }
}
#endif
#if VARIANT == VARIANT_OUTPTR
void outptr_combine_vec256(void(*f)(__m256, __m256, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use outptr VEC256 version\n"));
  for (int i = 0; i < n; i += 8) {
    __m256 a, b, c;
    a = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&B[i]));
    f(a, b, (float*)&c);
    _mm256_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m256, c));
  }
}
#endif
#if VARIANT == VARIANT_PTR
void ptr_combine_vec256(void(*f)(const float*, const float*, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ptr VEC256 version\n"));
  for (int i = 0; i < n; i += 8) {
    __m256 a, b, c;
    a = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m256, _mm256_loadu_ps(&B[i]));
    f((float*)&a, (float*)&b, (float*)&c);
    _mm256_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m256, c));
  }
}
#endif
#endif

#if defined(__AVX512F__)
#if VARIANT == VARIANT_REF
void ref_combine_vec512(float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ref VEC512 version\n"));
  for (int i = 0; i < n; i += 16) {
    __m512 a, b, c;
    a = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&B[i]));
    c = _mm512_add_ps(a, b);
    _mm512_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m512, c));
  }
}
#endif
#if VARIANT == VARIANT_VALUE
void value_combine_vec512(__m512(*f)(__m512, __m512), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use value VEC512 version\n"));
  for (int i = 0; i < n; i += 16) {
    __m512 a, b, c;
    a = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&B[i]));
    c = f(a, b);
    _mm512_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m512, c));
  }
}
#endif
#if VARIANT == VARIANT_INPTR
void inptr_combine_vec512(__m512(*f)(const float*, const float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use inptr VEC512 version\n"));
  for (int i = 0; i < n; i += 16) {
    __m512 a, b, c;
    a = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&B[i]));
    c = f((float*)&a, (float*)&b);
    _mm512_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m512, c));
  }
}
#endif
#if VARIANT == VARIANT_OUTPTR
void outptr_combine_vec512(void(*f)(__m512, __m512, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use outptr VEC512 version\n"));
  for (int i = 0; i < n; i += 16) {
    __m512 a, b, c;
    a = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&B[i]));
    f(a, b, (float*)&c);
    _mm512_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m512, c));
  }
}
#endif
#if VARIANT == VARIANT_PTR
void ptr_combine_vec512(void(*f)(const float*, const float*, float*), float* A, const float* B, int n) {
  VERBOSE(fprintf(stderr, "Use ptr VEC512 version\n"));
  for (int i = 0; i < n; i += 16) {
    __m512 a, b, c;
    a = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&A[i]));
    b = OPTIMIZATION_BARRIER(__m512, _mm512_loadu_ps(&B[i]));
    f((float*)&a, (float*)&b, (float*)&c);
    _mm512_storeu_ps(&A[i], OPTIMIZATION_BARRIER(__m512, c));
  }
}
#endif
#endif

#if VARIANT == VARIANT_REF
void ref_add_vec(int repeat, float* A, const float* B, int n) {
  for (int r = 0; r < repeat; ++r) {
#if defined(__AVX512F__)
    ref_combine_vec512(A, B, n);
#elif defined(__AVX__)
    ref_combine_vec256(A, B, n);
#elif defined(__SSE2__)
    ref_combine_vec128(A, B, n);
#else
    ref_combine_vec(A, B, n);
#endif
  }
}
#endif

#if VARIANT == VARIANT_VALUE
void value_add_vec(int repeat, float* A, const float* B, int n) {
  for (int r = 0; r < repeat; ++r) {
    if (0) {
#if defined(__AVX512F__)
    } else if (value_add512) {
      value_combine_vec512(value_add512, A, B, n);
#endif
#if defined(__AVX__)
    } else if (value_add256) {
      value_combine_vec256(value_add256, A, B, n);
#endif
#if defined(__SSE2__)
    } else if (value_add128) {
      value_combine_vec128(value_add128, A, B, n);
#endif
    } else if (value_add) {
      value_combine_vec(value_add, A, B, n);
    } else {
      fprintf(stderr, "No version of `add` available\n");
      fflush(stderr);
      abort();
    }
  }
}
#endif

#if VARIANT == VARIANT_INPTR
void inptr_add_vec(int repeat, float* A, const float* B, int n) {
  for (int r = 0; r < repeat; ++r) {
    if (0) {
#if defined(__AVX512F__)
    } else if (inptr_add512) {
      inptr_combine_vec512(inptr_add512, A, B, n);
#endif
#if defined(__AVX__)
    } else if (inptr_add256) {
      inptr_combine_vec256(inptr_add256, A, B, n);
#endif
#if defined(__SSE2__)
    } else if (inptr_add128) {
      inptr_combine_vec128(inptr_add128, A, B, n);
#endif
    } else if (inptr_add) {
      inptr_combine_vec(inptr_add, A, B, n);
    } else {
      fprintf(stderr, "No version of `add` available\n");
      fflush(stderr);
      abort();
    }
  }
}
#endif

#if VARIANT == VARIANT_OUTPTR
void outptr_add_vec(int repeat, float* A, const float* B, int n) {
  for (int r = 0; r < repeat; ++r) {
    if (0) {
#if defined(__AVX512F__)
    } else if (outptr_add512) {
      outptr_combine_vec512(outptr_add512, A, B, n);
#endif
#if defined(__AVX__)
    } else if (outptr_add256) {
      outptr_combine_vec256(outptr_add256, A, B, n);
#endif
#if defined(__SSE2__)
    } else if (outptr_add128) {
      outptr_combine_vec128(outptr_add128, A, B, n);
#endif
    } else if (outptr_add) {
      outptr_combine_vec(outptr_add, A, B, n);
    } else {
      fprintf(stderr, "No version of `add` available\n");
      fflush(stderr);
      abort();
    }
  }
}
#endif

#if VARIANT == VARIANT_PTR
void ptr_add_vec(int repeat, float* A, const float* B, int n) {
  for (int r = 0; r < repeat; ++r) {
    if (0) {
#if defined(__AVX512F__)
    } else if (ptr_add512) {
      ptr_combine_vec512(ptr_add512, A, B, n);
#endif
#if defined(__AVX__)
    } else if (ptr_add256) {
      ptr_combine_vec256(ptr_add256, A, B, n);
#endif
#if defined(__SSE2__)
    } else if (ptr_add128) {
      ptr_combine_vec128(ptr_add128, A, B, n);
#endif
    } else if (ptr_add) {
      ptr_combine_vec(ptr_add, A, B, n);
    } else {
      fprintf(stderr, "No version of `add` available\n");
      fflush(stderr);
      abort();
    }
  }
}
#endif

void add_vec(int repeat, float* A, const float* B, int n) {
#if VARIANT == VARIANT_REF
  ref_add_vec(repeat, A, B, n);
#endif
#if VARIANT == VARIANT_VALUE
  value_add_vec(repeat, A, B, n);
#endif
#if VARIANT == VARIANT_INPTR
  inptr_add_vec(repeat, A, B, n);
#endif
#if VARIANT == VARIANT_OUTPTR
  outptr_add_vec(repeat, A, B, n);
#endif
#if VARIANT == VARIANT_PTR
  ptr_add_vec(repeat, A, B, n);
#endif
}

int main() {
  int tries = 10000;
  int repeat = 100;
  int n = 1000;
  void* mem1 = malloc(n*sizeof(float) + 4096);
  void* mem2 = malloc(n*sizeof(float) + 4096);
  if (!mem1 || !mem2) {
    fprintf(stderr, "Failed to allocate 2x%d floats\n", n);
    fflush(stderr);
    abort();
  }
  uintptr_t imem1 = (((uintptr_t)mem1) + 4095) & -4096;
  uintptr_t imem2 = (((uintptr_t)mem2) + 4095) & -4096;
  float* A = (float*)imem1;
  float* B = (float*)imem2;

  for (int i = 0; i < n; ++i) {
    A[i] = i;
    B[i] = 1;
  }

  uint64_t dmin = -1;

  for (int t = 0; t < tries; ++t) {
    uint64_t t0 = _rdtsc();
    add_vec(repeat, A, B, n);
    uint64_t t1 = _rdtsc();
    uint64_t d = t1 - t0;
    dmin = d < dmin ? d : dmin;
  }

  printf("perf: %g cpp\n", (double)dmin  / (double)(n * repeat));

  free(mem2);
  free(mem1);
}
