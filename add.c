#include "add.h"

float _value_add(float a, float b) {
  return a + b;
}
float (*value_add)(float, float) = &_value_add;
float _inptr_add(const float* a, const float* b) {
  return *a + *b;
}
float (*inptr_add)(const float*, const float*) = &_inptr_add;
void _outptr_add(float a, float b, float* dst) {
  *dst = a + b;
}
void (*outptr_add)(float, float, float*) = &_outptr_add;
void _ptr_add(const float* a, const float* b, float* dst) {
  *dst = *a + *b;
}
void (*ptr_add)(const float*, const float*, float*) = &_ptr_add;
