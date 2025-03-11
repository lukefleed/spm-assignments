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
void softmax_auto_parallel(const float *input, float *output, size_t K,
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
void softmax_auto_noparallel(const float *input, float *output, size_t K) {
  float max_val = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : max_val)
  for (size_t i = 0; i < K; ++i) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
#pragma omp simd
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
  }
}


/**
 * @brief Wrapper function that delegates to either parallel or non-parallel implementation
 * based on compile-time configuration.
 *
 * This function provides a unified interface for the softmax computation while allowing
 * compile-time selection between parallel and non-parallel implementations. The selection
 * is controlled via the PARALLEL macro, which can be defined during compilation:
 * - When PARALLEL=0: Uses the non-parallel version optimized for small input sizes
 * - When PARALLEL=1: Uses the parallel version optimized for large input sizes
 *
 * This design pattern enables:
 * 1. Simple benchmarking between implementations without code changes
 * 2. Optimal performance across different workload sizes
 * 3. Single entry point for client code, hiding implementation details
 *
 * @param input Pointer to the input array (must be 32-byte aligned)
 * @param output Pointer to the output array (must be 32-byte aligned)
 * @param K Size of the input and output arrays
 * @param num_threads Number of threads to use (default: -1, use system default)
 */
void softmax_auto(const float *input, float *output, size_t K, int num_threads = -1) {
#if PARALLEL == 0
  softmax_auto_noparallel(input, output, K);
#else
  softmax_auto_parallel(input, output, K, num_threads);
#endif
}

// --------------------------------------------------------------------------//
// This code implementation includes a standalone benchmarking mechanism with a
// main function that allows direct timing measurement of the softmax
// implementations. While you're supposed to use `make test`
// for formal benchmarking, this approach offers an alternative that directly
// prints the elapsed time using the TIMERSTART and TIMERSTOP macros from the
// original code.
// --------------------------------------------------------------------------//

/**
 * @brief Custom C++17 aligned memory allocator for SIMD operations
 *
 * This allocator leverages C++17's aligned memory allocation features to ensure
 * proper memory alignment for optimal SIMD performance. While auto-vectorization
 * can work with unaligned memory, aligned memory access significantly improves
 * performance by:
 *
 * 1. Eliminating the need for unaligned load/store instructions
 * 2. Avoiding potential cache-line splits during vector operations
 * 3. Preventing performance penalties on architectures with strict alignment requirements
 *
 * The 32-byte alignment is selected to accommodate AVX/AVX2 instructions (256-bit vectors),
 * ensuring optimal performance across different SIMD instruction sets.
 *
 * @tparam T The type of elements to allocate
 */
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment required for AVX

  /**
   * @brief Allocate memory using the C++17 aligned operator new
   *
   * This method allocates memory with the specified alignment using C++17's
   * aligned allocation feature. Unlike older custom alignment techniques,
   * this approach is standard-compliant and platform-independent.
   *
   * @param n Number of elements to allocate
   * @return T* Pointer to aligned memory block or nullptr if n is zero
   */
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  /**
   * @brief Deallocate aligned memory
   *
   * Uses the C++17 aligned operator delete to ensure proper deallocation
   * of the previously aligned memory. This prevents memory leaks and ensures
   * that the correct deallocation function is called.
   *
   * @param p Pointer to the memory block to deallocate
   * @param n Size of allocation (required by allocator interface but unused)
   */
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

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocatorC17<T>>;

#ifndef TEST_BUILD

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
/**
 * @brief Main function for standalone benchmarking
 *
 * Provides a command-line interface for testing the auto-vectorized softmax implementation.
 * Usage:
 * - First argument: Size of input array (K)
 * - Second argument: Optional flag to print results (any value works)
 *
 * The runtime measurements use the TIMERSTART/TIMERSTOP macros from hpc_helpers.hpp,
 * which provide high-precision timing using platform-specific timers. This offers
 * an alternative benchmarking approach to the standard `make test` method.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return int Exit code (0 for success)
 */
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
