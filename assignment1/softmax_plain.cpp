#include <algorithm>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

/**
 * @brief Computes the softmax function for a given input array.
 *
 * This implementation uses numerical stabilization by subtracting the maximum
 * value from all inputs before computing exponentials to prevent overflow.
 *
 * The softmax function is defined as:
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))) for all j
 *
 * @param input Pointer to the input array containing K floating-point values
 * @param output Pointer to the output array where softmax results will be
 stored.
 *               Must be pre-allocated with at least K elements.
 * @param K The number of elements in both input and output arrays
 *
 * @note The input and output arrays can be the same (in-place operation)
 * @note This function assumes both input and output pointers are valid and
 *       point to arrays of at least K elements
 */
void softmax_plain(const float *input, float *output, size_t K) {
  // Find the maximum to stabilize the computation of the exponential
  float max_val = -std::numeric_limits<float>::max();
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

// --------------------------------------------------------------------------//
// This code implementation includes a standalone benchmarking mechanism with a
// main function that allows direct timing measurement of the softmax
// implementations. While you're supposed to use `make test`
// for formal benchmarking, this approach offers an alternative that directly
// prints the elapsed time using the TIMERSTART and TIMERSTOP macros from the
// original code.
// --------------------------------------------------------------------------//

#ifndef TEST_BUILD
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

  TIMERSTART(softime_plain);
  softmax_plain(input.data(), output.data(), K);
  TIMERSTOP(softime_plain);

  // print the results on the standard output
  if (print) {
    printResult(output, K);
  }
}
#endif
