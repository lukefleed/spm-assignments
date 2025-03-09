#include <algorithm>
#include <cmath>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <new> // For std::align_val_t
#include <random>
#include <vector>

void softmax_auto(const float *input, float *output, size_t K) {
  float max_val = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : max_val)
  // #pragma omp parallel for simd reduction(max : max_val)
  for (size_t i = 0; i < K; ++i) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
  // #pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
#pragma omp simd
  // #pragma omp parallel for simd
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
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

// This allocator uses the over-aligned new and delete operators provided in
// C++17 to guarantee that allocated memory is aligned to 32 bytes (suitable for
// AVX).
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment required for AVX

  // Allocate memory using the C++17 aligned operator new.
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  // Deallocate memory using the C++17 aligned operator delete.
  void deallocate(T *p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(alignment));
  }

  template <typename U>
  bool operator==(const AlignedAllocatorC17<U> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocatorC17<U> &) const noexcept {
    return false;
  }
};

// Alias for a vector using the C++17 aligned allocator.
template <typename T>
using aligned_vector = std::vector<T, AlignedAllocatorC17<T>>;

// Generate random input data with a fixed seed
aligned_vector<float> generate_random_input(size_t K, float min = -1.0f,
                                            float max = 1.0f) noexcept {
  aligned_vector<float> input(K);
  std::mt19937 gen(5489); // Fixed seed for reproducible results
  std::uniform_real_distribution<float> dis(min, max);

  for (size_t i = 0; i < K; ++i) {
    input[i] = dis(gen);
  }

  return input;
}

void printResult(const aligned_vector<float> &v, size_t K) {
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

  // Generate aligned random data
  aligned_vector<float> input = generate_random_input(K);
  aligned_vector<float> output(K);

  // Benchmark auto-vectorized implementation
  TIMERSTART(softmax_auto);
  softmax_auto(input.data(), output.data(), K);
  TIMERSTOP(softmax_auto);

  // Print results if requested
  if (print) {
    printResult(output, K);
  }

  return 0;
}
