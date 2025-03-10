#include <algorithm>
#include <cmath>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <new>
#include <omp.h>
#include <random>
#include <vector>

/**
 * @brief Compute the softmax of an input array using auto-vectorization.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 * @param num_threads Number of threads to use (default is -1, which means no
 * change).
 *
 * This function computes the softmax of the input array using
 * auto-vectorization Consider removing the current pragma omp simd and
 * substituting it with the commented one. For small values of K, the overhead
 * of OpenMP is not worth it.
 *
 * ## Auto-Vectorized Implementation Details
 *
 * This implementation includes several optimizations:
 * - `#pragma omp simd` directives vectorize the main computational loops
 * - Reduction clauses ensure correct maximum value and sum calculations
 * - Using `expf()` instead of `std::exp()` improves SIMD performance
 * - Precomputing the inverse sum (`inv_sum = 1.0f / sum`) and using
 *   multiplication instead of division improves efficiency
 * - Explicit comparisons replace `std::max()` to aid vectorization
 *
 * ## Performance
 *
 * Using parallelization for small K values is not convenient. The threads
 * will not yield a performance benefit due to overhead. This will result
 * in a performance drop compared to the scalar version.
 *
 * For large K values, the performance will be similar to the manual
 * vectorized version if the machine supports AVX512 and the flag
 * `-march=native` is used.
 */
void softmax_auto(const float *input, float *output, size_t K,
                  int num_threads = -1) {
  // Set thread count if specified
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

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

/// The following code does not use parallelization, suggest to use it for
/// small K values.

// void softmax_auto(const float *input, float *output, size_t K) {
//   float max_val = -std::numeric_limits<float>::infinity();
// #pragma omp simd reduction(max : max_val)
//   for (size_t i = 0; i < K; ++i) {
//     if (input[i] > max_val) {
//       max_val = input[i];
//     }
//   }

//   float sum = 0.0f;
// #pragma omp simd reduction(+ : sum)
//   for (size_t i = 0; i < K; ++i) {
//     output[i] = expf(input[i] - max_val);
//     sum += output[i];
//   }

//   const float inv_sum = 1.0f / sum;
// #pragma omp simd
//   for (size_t i = 0; i < K; ++i) {
//     output[i] *= inv_sum;
//   }
// }

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

#ifndef TEST_BUILD
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
#endif

#ifndef TEST_BUILD
int main(int argc, char *argv[]) {
  int num_threads = -1; // Default: use system default

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
  softmax_auto(input.data(), output.data(), K, num_threads);
  TIMERSTOP(softmax_auto);

  // Print results if requested
  if (print) {
    printResult(output, K);
  }

  return 0;
}
#endif
