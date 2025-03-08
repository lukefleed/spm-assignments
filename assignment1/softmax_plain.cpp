#include <algorithm>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

void softmax_plain(const float *input, float *output, size_t K) {
  // Find the maximum to stabilize the computation of the exponential
  float max_val = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // computes all exponentials with the shift of max_val and the total sum
  float sum = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  // normalize by dividing for the total sum
  for (size_t i = 0; i < K; ++i) {
    output[i] /= sum;
  }
}
