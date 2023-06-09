# Interflop

## SIMD

Example: `A[:] += B[:]`

### reference output

```asm
  401a40: movslq  %eax,%r8
  401a43: add     $0x8,%eax
  401a46: lea     (%rdi,%r8,4),%rcx
  401a4a: vmovups (%rsi,%r8,4),%ymm1
  401a50: vmovups (%rcx),%ymm0
  401a54: vaddps  %ymm1,%ymm0,%ymm0
  401a58: vmovups %ymm0,(%rcx)
  401a5c: cmp     %eax,%edx
  401a5e: jg      401a40 <ref_combine_vec256+0x10>
```

### value interface

#### interface
```c
float  (*add   )(float , float ); // xmm0, xmm1 -> xmm0
__m128 (*add128)(__m128, __m128); // xmm0, xmm1 -> xmm0
#ifdef __AVX__
__m256 (*add256)(__m256, __m256); // ymm0, ymm1 -> ymm0
#endif
```

#### Assembly output

Caller:
```asm
  // Same as ref except vaddps has been replaced with a call
  401aa8: movslq  %ebx,%rdx
  401aab: add     $0x8,%ebx
  401aae: lea     (%r14,%rdx,4),%r12
  401ab2: vmovups (%r15,%rdx,4),%ymm1
  401ab8: vmovups (%r12),%ymm0
  401abe: call    *%r13
  401ac1: vmovups %ymm0,(%r12)
  401ac7: cmp     %ebx,-0x34(%rbp)
  401aca: jg      401aa8 <value_combine_vec256+0x38>
```


Callee:
```asm
0000000000402520 <_value_add256>:
  402520: vaddps %ymm1,%ymm0,%ymm0
  402524: ret
```

### in ptr

#### Interface

```c
float  (*add   )(const float*, const float*); // rdi[1], rsi[1] -> xmm0
__m128 (*add128)(const float*, const float*); // rdi[4], rsi[4] -> xmm0
#ifdef __AVX__
__m256 (*add256)(const float*, const float*); // rdi[8], rsi[8] -> ymm0
#endif
```

#### Assembly output

Caller:
```asm
  // A bit of stack fiddling is required to call with the right ABI
  401b40: movslq  %r15d,%rcx
  401b43: mov     0x8(%rsp),%rsi
  401b48: mov     0x10(%rsp),%rdi
  401b4d: add     $0x8,%r15d
  401b51: lea     0x0(%r13,%rcx,4),%rbx
  401b56: vmovups (%rbx),%ymm0
  401b5a: vmovaps %ymm0,0x20(%rsp)
  401b60: vmovups (%r14,%rcx,4),%ymm0
  401b66: vmovaps %ymm0,0x40(%rsp)
  401b6c: call    *%r12
  401b6f: vmovups %ymm0,(%rbx)
  401b73: cmp     %r15d,0x1c(%rsp)
  401b78: jg      401b40 <inptr_combine_vec256+0x50>
```

Callee:
```asm
0000000000402530 <_inptr_add256>:
  402530: vmovups (%rdi),%ymm0
  402534: vaddps  (%rsi),%ymm0,%ymm0
  402538: ret
```

### out ptr

#### Interface

```c
void (*add   )(float , float , float*); // xmm0, xmm1 -> rax[1]
void (*add128)(__m128, __m128, float*); // xmm0, xmm1 -> rax[4]
#ifdef __AVX__
void (*add256)(__m256, __m256, float*); // xmm0, xmm1 -> rax[8]
#endif
```

#### Assembly output

```asm
  // Also a bit of stack fiddling
  401c00: movslq  %r15d,%rcx
  401c03: mov     -0x80(%rbp),%rdi
  401c07: add     $0x8,%r15d
  401c0b: lea     0x0(%r13,%rcx,4),%rbx
  401c10: vmovups (%r14,%rcx,4),%ymm1
  401c16: vmovups (%rbx),%ymm0
  401c1a: call    *%r12
  401c1d: vmovaps -0x70(%rbp),%ymm0
  401c22: vmovups %ymm0,(%rbx)
  401c26: cmp     %r15d,-0x74(%rbp)
  401c2a: jg      401c00 <outptr_combine_vec256+0x50>
```

Callee:
```asm
0000000000402540 <_outptr_add256>:
  402540: vaddps  %ymm1,%ymm0,%ymm0
  402544: vmovups %ymm0,(%rdi)
  402548: ret
```

### in/out ptr

#### Interface

```c
void (*add   )(const float*, const float*, float*); // rdi[1], rsi[1] -> rax[1]
void (*add128)(const float*, const float*, float*); // rdi[4], rsi[4] -> rax[4]
// SIMD does not appear in ABI of this function
// It could be implemented using scalar or SSE
void (*add256)(const float*, const float*, float*); // rdi[8], rsi[8] -> rax[8]
```
#### Assembly output

Caller:
```asm
  // Many moves to adjust the stack in the right way
  // A `vzeroupper` to accomodate the fact that ABI is not SIMD
  401cc0: movslq  %r15d,%rdx
  401cc3: mov     0x10(%rsp),%rsi
  401cc8: mov     (%rsp),%rdi
  401ccc: lea     0x0(%r13,%rdx,4),%rbx
  401cd1: vmovups (%rbx),%ymm0
  401cd5: vmovaps %ymm0,0x20(%rsp)
  401cdb: vmovups (%r14,%rdx,4),%ymm0
  401ce1: mov     0x8(%rsp),%rdx
  401ce6: vmovaps %ymm0,0x40(%rsp)
  401cec: vzeroupper // callee might use plain old SSE instructions
  401cef: call    *%r12
  401cf2: add     $0x8,%r15d
  401cf6: vmovaps 0x60(%rsp),%ymm0
  401cfc: vmovups %ymm0,(%rbx)
  401d00: cmp     %r15d,0x1c(%rsp)
  401d05: jg      401cc0 <ptr_combine_vec256+0x60>
```

Callee:
```asm
0000000000402550 <_ptr_add256>:
  402550: vmovups (%rdi),%ymm0
  402554: vaddps  (%rsi),%ymm0,%ymm0
  402558: vmovups %ymm0,(%rdx)
  40255c: vzeroupper // caller might use plain old SSE instructions
  40255f: ret
```


### How to compile front and back with different ISA support?

```c
/* add.h */
extern float (*add)(float, float);
extern __m128 (*add128)(__m128, __m128);
#ifdef __AVX__
extern __m256 (*add256)(__m256, __m256);
#endif
```

```c
/* add.avx.c */
#include "add.h"

#ifdef __AVX__
#include <ymmintrin.h>

__m256 _add256(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}
__m256 (*add256)(__m256, __m256) = &_add256;
#else
// Set the symbol `add256` to NULL to notify the frontend that it is not supported
// This could be done with weak symbol to simplify the code a bit
void (*add256)() = NULL;
#endif
```


```c
/* main.c */
#include "add.h"

void combine_vec   (float (*)(float , float ), float* A, const float* B, int n);
void combine_vec128(__m128(*)(__m128, __m128), float* A, const float* B, int n);
#ifdef __AVX__
void combine_vec256(__m256(*)(__m256, __m256), float* A, const float* B, int n);
#endif

void add_vec(float* A, const float* B, int n) {
    if (0) {
#ifdef __AVX__
    } else if (add256) {
        combine_vec256(add256, A, B, n);
#endif
    } else if (add128) {
        combine_vec128(add128, A, B, n);
    } else if (add) {
        combine_vec(add, A, B, n);
    } else {
        // no variant found
        abort();
    }
}
```

With this design, `main.c` and `add*.c` could be compiled targeting different architectures and everything will work as expected.

### Benchmark

#### Sum

Code benched (arrays fit in L1):
```
A[0:1000] += B[0:1000]
```

Time measurment in cycles per point (cpp).
`ref` is a version without indirect call.


| variant | scalar |  sse |  avx | avx512 |
|:--------|:------:|:----:|:----:|:------:|
| ref     |  0.79  | 0.21 | 0.12 |  0.08  |
| value   |  2.33  | 0.59 | 0.37 |  0.19  |
| inptr   |  2.90  | 0.74 | 0.38 |  0.23  |
| outptr  |  2.33  | 0.59 | 0.38 |  0.19  |
| ptr     |  2.92  | 0.89 | 0.53 |  0.94  |

All variants give roughly the same performance (up to quality of the optimizations).
We should need to test on codes that have more operations to see if this stays true.
The assembly shows that the value variant is very efficient in term of instructions.

Even though the ptr variant would have been nice for the compatibility, it does not play well with the machine.

#### Fibonacci

Code benched (arrays fit in L1):
```
A[0:1000] = fibonacci(A[0:1000], B[0:1000], 64)
```

Time measurment in cycles per point (cpp).
`ref` is a version without indirect call.


| variant | scalar |  sse |  avx | avx512 |
|:--------|:------:|:----:|:----:|:------:|
| ref     |  0.73  | 0.18 | 0.09 |  0.05  |
| value   |  2.15  | 0.54 | 0.28 |  0.14  |
| inptr   |  4.23  | 1.05 | 0.58 |  0.31  |
| outptr  |  4.01  | 1.01 | 0.55 |  0.29  |
| ptr     |  6.97  | 1.74 | 0.97 |  1.17  |

When the Arithmetic Intensity is higher (less memory access per operation), the difference between the variants are larger.
In particular, the pass-by value variant is twice faster than inptr and outptr, and 3-4x faster than ptr, and even 8x faster in AVX512.
