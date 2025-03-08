#include <algorithm>
#include <avx_mathfun.h>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

void softmax_avx(const float *input, float *output, size_t K) {
  // Find the maximum to stabilize the computation of the exponential
  // First using AVX to find local maximums
  float max_val = -std::numeric_limits<float>::infinity();
  size_t i = 0;

  // Process 8 floats at a time using AVX
  if (K >= 8) {
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    for (; i + 7 < K; i += 8) {
      __m256 input_vec = _mm256_loadu_ps(input + i);
      max_vec = _mm256_max_ps(max_vec, input_vec);
    }

    // Extract the maximum value from the AVX vector
    float max_array[8] __attribute__((aligned(32)));
    _mm256_store_ps(max_array, max_vec);

    for (int j = 0; j < 8; ++j) {
      max_val = std::max(max_val, max_array[j]);
    }
  }

  // Process remaining elements
  for (; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // Create a vector with the maximum value
  __m256 max_vec = _mm256_set1_ps(max_val);

  // Compute the exponentials and sum them
  __m256 sum_vec = _mm256_setzero_ps();
  i = 0;

  // Process groups of 8 elements
  for (; i + 7 < K; i += 8) {
    // Load input values
    __m256 input_vec = _mm256_loadu_ps(input + i);

    // Subtract max value for numerical stability
    __m256 shifted_vec = _mm256_sub_ps(input_vec, max_vec);

    // Calculate exponential using the provided exp256_ps function
    __m256 exp_vec = exp256_ps(shifted_vec);

    // Store result in output array
    _mm256_storeu_ps(output + i, exp_vec);

    // Add to sum
    sum_vec = _mm256_add_ps(sum_vec, exp_vec);
  }

  // Process remaining elements and add to sum
  float sum = 0.0f;

  // Extract the sum from the AVX vector
  float sum_array[8] __attribute__((aligned(32)));
  _mm256_store_ps(sum_array, sum_vec);

  for (int j = 0; j < 8; ++j) {
    sum += sum_array[j];
  }

  // Process remaining elements sequentially
  for (; i < K; ++i) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  // Normalize by dividing by the sum
  __m256 sum_reciprocal_vec = _mm256_set1_ps(1.0f / sum);
  i = 0;

  // Process groups of 8 elements
  for (; i + 7 < K; i += 8) {
    __m256 output_vec = _mm256_loadu_ps(output + i);
    __m256 normalized_vec = _mm256_mul_ps(output_vec, sum_reciprocal_vec);
    _mm256_storeu_ps(output + i, normalized_vec);
  }

  // Process remaining elements
  for (; i < K; ++i) {
    output[i] /= sum;
  }
}

std::vector<float> generate_random_input(size_t K, float min = -1.0f,
                                         float max = 1.0f) {
  std::vector<float> input(K);
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(5489); // fixed seed for reproducible results
  std::uniform_real_distribution<float> dis(min, max);
  for (size_t i = 0; i < K; ++i) {
    input[i] = dis(gen);
  }
  return input;
}

void printResult(std::vector<float> &v, size_t K) {
  for (size_t i = 0; i < K; ++i) {
    std::fprintf(stderr, "%f\n", v[i]);
  }
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::printf("use: %s K [1]\n", argv[0]);
    return 0;
  }
  size_t K = 0;
  if (argc >= 2) {
    K = std::stol(argv[1]);
  }
  bool print = false;
  if (argc == 3) {
    print = true;
  }
  std::vector<float> input = generate_random_input(K);
  std::vector<float> output(K);

  TIMERSTART(softime_avx);
  softmax_avx(input.data(), output.data(), K);
  TIMERSTOP(softime_avx);

  // print the results on the standard output
  if (print) {
    printResult(output, K);
  }
}
