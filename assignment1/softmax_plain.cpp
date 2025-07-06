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
 * value from all inputs before computing the exponential to prevent overflow.
 *
 * @param input Pointer to the input array of size K containing the raw scores
 * @param output Pointer to the output array of size K where softmax
 * probabilities will be stored
 * @param K The number of elements in both input and output arrays
 *
 * @note The output array must be pre-allocated with at least K elements
 * @note The function modifies the output array in-place
 * @note Input and output arrays should not overlap
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
