#include <algorithm>
#include <cmath>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <new>
#include <omp.h>
#include <random>
#include <vector>

// Determine vector width and alignment based on available instruction sets
#ifdef __AVX512F__
#define VECTOR_ALIGNMENT 64    // AVX512 requires 64-byte alignment
#define ELEMENTS_PER_VECTOR 16 // 16 floats per 512-bit register
#else
#define VECTOR_ALIGNMENT 32   // AVX2 requires 32-byte alignment
#define ELEMENTS_PER_VECTOR 8 // 8 floats per 256-bit register
#endif

/**
 * @brief Custom C++17 aligned memory allocator for SIMD operations
 *
 * Uses C++17's aligned memory features to ensure proper alignment
 * for optimal SIMD performance. Aligned memory access improves performance by:
 * - Eliminating unaligned load/store instructions
 * - Avoiding cache-line splits
 * - Preventing penalties on architectures with strict alignment requirements
 *
 * The alignment is automatically determined based on available instruction
 * sets:
 * - AVX512: 64-byte alignment (512-bit registers)
 * - AVX2: 32-byte alignment (256-bit registers)
 *
 * @tparam T The type of elements to allocate
 */
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment =
      VECTOR_ALIGNMENT; // Use dynamic alignment based on available instructions

  /**
   * @brief Allocate aligned memory using C++17 features
   * @param n Number of elements to allocate
   * @return T* Pointer to aligned memory or nullptr if n is zero
   */
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  /**
   * @brief Deallocate aligned memory
   * @param p Pointer to memory block
   * @param n Size (required by allocator interface)
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

/**
 * @brief Compute softmax with parallelism and auto-vectorization
 *
 * @param input Aligned input array pointer
 * @param output Aligned output array pointer
 * @param K Size of arrays
 * @param num_threads Thread count (-1 for system default)
 */
void softmax_auto_parallel(const float *__restrict__ input,
                           float *__restrict__ output, size_t K,
                           int num_threads = -1) {
  /**
   * OPTIMIZATION NOTES:
   * - __restrict__ qualifier: Informs compiler pointers don't alias
   * - Ternary operator: Enables better vectorization vs if-statements
   * - Separate loops: Better vectorization and cache utilization
   * - expf() instead of std::exp(): Faster single-precision SIMD operations
   * - Multiplication by inverse: Faster than repeated divisions
   */

  // Set thread count if specified
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  float max_val = -std::numeric_limits<float>::infinity();

#pragma omp parallel for simd reduction(max : max_val)                         \
    aligned(input : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    max_val = (input[i] > max_val) ? input[i] : max_val;
  }

  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)                               \
    aligned(input, output : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
#pragma omp parallel for simd aligned(output : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
  }
}

/**
 * @brief Non-parallel softmax with auto-vectorization
 *
 * Uses SIMD instructions but no threading - better for small K values
 * where thread creation overhead exceeds parallelization benefits.
 *
 * @param input Aligned input array pointer
 * @param output Aligned output array pointer
 * @param K Size of arrays
 */
void softmax_auto_noparallel(const float *__restrict__ input,
                             float *__restrict__ output, size_t K) {
  float max_val = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : max_val) aligned(input : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    max_val = (input[i] > max_val) ? input[i] : max_val;
  }

  float sum = 0.0f;
#pragma omp simd reduction(+ : sum) aligned(input, output : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  const float inv_sum = 1.0f / sum;
#pragma omp simd aligned(output : VECTOR_ALIGNMENT)
  for (size_t i = 0; i < K; ++i) {
    output[i] *= inv_sum;
  }
}

/**
 * @brief Unified interface for softmax computation
 *
 * Selects implementation based on PARALLEL macro:
 * - PARALLEL=0: Non-parallel version for small inputs
 * - PARALLEL=1: Parallel version for large inputs
 *
 * @param input Aligned input array pointer
 * @param output Aligned output array pointer
 * @param K Size of arrays
 * @param num_threads Thread count (-1 for system default)
 */
void softmax_auto(const float *input, float *output, size_t K,
                  int num_threads = -1) {
#if PARALLEL == 0
  softmax_auto_noparallel(input, output, K);
#else
  softmax_auto_parallel(input, output, K, num_threads);
#endif
}

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
 * @brief Standalone benchmarking interface
 *
 * Usage: program K [print_flag]
 * - K: Size of input array
 * - print_flag: Optional flag to print results
 *
 * Uses TIMERSTART/TIMERSTOP macros for high-precision timing as an
 * alternative to the standard `make test` benchmarking approach.
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

  // Display alignment information
#ifdef __AVX512F__
  std::printf("Using AVX512 with 64-byte alignment\n");
#else
  std::printf("Using AVX2 with 32-byte alignment\n");
#endif

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
