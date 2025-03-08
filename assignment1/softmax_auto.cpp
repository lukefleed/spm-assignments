#include <algorithm>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

void softmax_auto(const float *input, float *output, size_t K) {
  if (K == 0)
    return;

  float max_val = input[0];
  for (size_t i = 1; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  float sum = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
  }
}
