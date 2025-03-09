#include <algorithm>
#include <cmath>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

void softmax_auto(const float *input, float *output, size_t K) {
  float max_val = -std::numeric_limits<float>::infinity();
#pragma omp parallel for simd reduction(max : max_val)
  for (size_t i = 0; i < K; ++i) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
#pragma omp parallel for simd
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
  }
}
