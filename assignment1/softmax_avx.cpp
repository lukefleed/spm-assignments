#include <algorithm>
#include <avx_mathfun.h>
#include <hpc_helpers.hpp>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

/**
 *
 * @brief Manually Vectorized Softmax Implementation
 *
 * This implementation employs a three-phase approach with explicit AVX2
 * intrinsics to achieve maximum performance:
 *
 * 1. Find maximum value across the input array
 * 2. Compute exponentials and sum
 * 3. Normalize by the sum
 *
 * Key optimizations:
 * - Loop unrolling (4x for processing 32 elements at once)
 * - Software prefetching
 * - Efficient horizontal reduction patterns
 * - Principled masking approach via `compute_mask()` to handle non-multiples of
 * 8 (AVX register width) without requiring a separate remainder loop
 * - Cache blocking with a 32KB block size to minimize L1 cache misses during
 *   multi-phase processing
 *
 * Parallelization strategy:
 * - OpenMP parallelization across available hardware threads
 * - Standard `#pragma omp parallel for reduction(max:max_val)` for maximum
 * finding
 * - Custom approach for sum calculation with manual local reductions and atomic
 *   updates to minimize false sharing and synchronization overhead
 * - Specialized variant (`softmax_avx_small`) for small inputs that avoids
 *   OpenMP threading overhead while maintaining AVX optimizations
 */

/**
 * @brief Helper function to generate a mask for remaining elements in a vector.
 * @param n Number of remaining elements (0 < n < 8).
 * @return __m256i mask where the first `n` elements are set to -1 (active), and
 * the rest are 0 (inactive).
 */
static inline __m256i compute_mask(size_t n) {
  const __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), indices);
}

/**
 * @brief AVX-accelerated softmax implementation with masking and OpenMP
 * parallelization.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 * @param num_threads Number of threads to use (default: -1, use all available).
 *
 * This function computes the softmax of the input array using AVX instructions,
 * OpenMP parallelization, and masking to handle non-multiple-of-8 elements
 * efficiently. It is optimized for large arrays.
 */
void softmax_avx(const float *input, float *output, size_t K,
                 int num_threads = -1) {
  const size_t BLOCK_SIZE =
      32 * 1024 / sizeof(float); // Block size for cache-friendly processing
                                 // (approximately 8K floats)
  float max_val =
      -std::numeric_limits<float>::infinity(); // Initialize overall maximum to
                                               // negative infinity

  // Use specified thread count or default to processor count
  int threads_to_use = (num_threads > 0) ? num_threads : omp_get_num_procs();

// Parallelize the max finding operation with reduction across all threads
#pragma omp parallel for reduction(max : max_val) num_threads(threads_to_use)
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    const size_t block_end =
        std::min(block_start + BLOCK_SIZE,
                 K); // Handle last block potentially being smaller
    __m256 max_vec = _mm256_set1_ps(
        -std::numeric_limits<float>::infinity()); // Initialize block maximum
                                                  // vector

    size_t i = block_start;
    // Process 32 elements per iteration (4x unrolling of AVX 8-float vectors)
    for (; i + 31 < block_end; i += 32) {
      _mm_prefetch(
          reinterpret_cast<const char *>(input + i + 128),
          _MM_HINT_T0); // Prefetch data 128 elements ahead into L1 cache
      // Load 4 AVX vectors (32 floats total) from input
      const __m256 data0 =
          _mm256_load_ps(input + i); // Loads 8 floats starting at position i
      const __m256 data1 = _mm256_load_ps(input + i + 8);  // Next 8 floats
      const __m256 data2 = _mm256_load_ps(input + i + 16); // Next 8 floats
      const __m256 data3 = _mm256_load_ps(input + i + 24); // Next 8 floats

      // Update max_vec by comparing with each data vector
      max_vec = _mm256_max_ps(max_vec, data0); // Element-wise maximum
      max_vec = _mm256_max_ps(max_vec, data1);
      max_vec = _mm256_max_ps(max_vec, data2);
      max_vec = _mm256_max_ps(max_vec, data3);
    }

    // Handle leftover elements in groups of 8
    for (; i + 7 < block_end; i += 8) {
      const __m256 data = _mm256_load_ps(input + i);
      max_vec = _mm256_max_ps(max_vec, data);
    }

    // Handle remaining elements (less than 8) using masking
    const size_t remaining = block_end - i;
    if (remaining > 0) {
      const __m256i mask = compute_mask(
          remaining); // Create mask where only valid elements are active
      const __m256 data = _mm256_maskload_ps(
          input + i, mask); // Masked load of remaining elements
      const __m256 blended = _mm256_blendv_ps(
          _mm256_set1_ps(-std::numeric_limits<float>::infinity()), data,
          _mm256_castsi256_ps(
              mask)); // Replace inactive elements with negative infinity
      max_vec =
          _mm256_max_ps(max_vec, blended); // Update max with the masked data
    }

    // Horizontal reduction to find the maximum value within the vector
    // First, swap high/low 128-bit lanes and compare
    __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec,
                                        0x01); // Swap high/low 128-bits
    max_vec = _mm256_max_ps(max_vec, tmp);
    // Then shuffle within 128-bit lanes and compare
    tmp = _mm256_shuffle_ps(
        max_vec, max_vec,
        _MM_SHUFFLE(1, 0, 3, 2)); // Shuffle within 128-bit lanes
    max_vec = _mm256_max_ps(max_vec, tmp);
    // Final shuffle and max to get the maximum in all positions
    tmp = _mm256_shuffle_ps(max_vec, max_vec,
                            _MM_SHUFFLE(2, 3, 0, 1)); // Shuffle again
    max_vec = _mm256_max_ps(max_vec, tmp);

    // Extract the maximum value from the first position of the vector
    const float block_max = _mm256_cvtss_f32(max_vec);
    max_val = std::max(max_val, block_max); // Update global maximum
    // Alternative using intrinsic: max_val = fmaxf(max_val, block_max);
  }

  // Phase 2: Compute exponentials and sum with masking
  float sum = 0.0f;
#pragma omp parallel num_threads(threads_to_use)
  {
    // Each thread maintains its own local sum to avoid false sharing
    float local_sum = 0.0f;
    // Broadcast max_val to all lanes of a vector register for efficient
    // subtraction
    const __m256 max_broadcast = _mm256_set1_ps(max_val);

    // Use nowait to allow threads to proceed to the atomic update without
    // waiting for all threads to finish the for loop
#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      // Use two accumulators to reduce dependency chains and improve
      // instruction-level parallelism
      __m256 sum0 = _mm256_setzero_ps(); // First accumulator vector (8 floats)
      __m256 sum1 = _mm256_setzero_ps(); // Second accumulator vector (8 floats)

      size_t i = block_start;
      // Process 32 elements per iteration (4 AVX vectors = 32 floats) for
      // better vectorization factor
      for (; i + 31 < block_end; i += 32) {
        const __m256 data0 = _mm256_load_ps(input + i); // Load first 8 floats
        const __m256 data1 =
            _mm256_load_ps(input + i + 8); // Load next 8 floats
        const __m256 data2 =
            _mm256_load_ps(input + i + 16); // Load next 8 floats
        const __m256 data3 =
            _mm256_load_ps(input + i + 24); // Load next 8 floats

        // Compute exp(x - max_val) for numerical stability (prevents overflow)
        const __m256 exp0 = exp256_ps(_mm256_sub_ps(data0, max_broadcast));
        const __m256 exp1 = exp256_ps(_mm256_sub_ps(data1, max_broadcast));
        const __m256 exp2 = exp256_ps(_mm256_sub_ps(data2, max_broadcast));
        const __m256 exp3 = exp256_ps(_mm256_sub_ps(data3, max_broadcast));

        // Store intermediate results for later normalization (phase 3)
        _mm256_store_ps(output + i, exp0);
        _mm256_store_ps(output + i + 8, exp1);
        _mm256_store_ps(output + i + 16, exp2);
        _mm256_store_ps(output + i + 24, exp3);

        // Accumulate sums using two accumulators to reduce dependency chains
        // and enable better instruction-level parallelism
        sum0 = _mm256_add_ps(
            sum0, _mm256_add_ps(exp0, exp1)); // Add first 16 elements to sum0
        sum1 = _mm256_add_ps(
            sum1, _mm256_add_ps(exp2, exp3)); // Add second 16 elements to sum1
      }

      // Handle leftover elements in groups of 8 (one AVX vector)
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_store_ps(output + i, exp);
        sum0 = _mm256_add_ps(sum0, exp); // Add to first accumulator
      }

      // Handle remaining elements (less than 8) using masking for correct
      // boundary handling
      const size_t remaining = block_end - i;
      if (remaining > 0) {
        const __m256i mask =
            compute_mask(remaining); // Create mask for valid elements only
        const __m256 data = _mm256_maskload_ps(input + i, mask); // Masked load
        const __m256 exp =
            exp256_ps(_mm256_sub_ps(data, max_broadcast)); // Compute exp
        _mm256_maskstore_ps(output + i, mask, exp);        // Masked store

        // Use blending to zero out invalid lanes before accumulating
        const __m256 blended = _mm256_blendv_ps(_mm256_setzero_ps(), exp,
                                                _mm256_castsi256_ps(mask));
        sum0 = _mm256_add_ps(sum0, blended); // Add masked values to accumulator
      }

      // Horizontal reduction to accumulate the sum across vector lanes
      __m256 sum_vec = _mm256_add_ps(sum0, sum1); // Combine both accumulators

      // Step 1: Swap high/low 128-bit lanes and add
      __m256 tmp =
          _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01); // Swap 128-bit lanes
      sum_vec = _mm256_add_ps(sum_vec, tmp); // Add corresponding elements

      // Step 2: Horizontal add within 128-bit lanes (adds adjacent pairs)
      tmp = _mm256_hadd_ps(sum_vec, sum_vec);

      // Step 3: Another horizontal add to get the final sum in the lowest
      // element
      sum_vec = _mm256_hadd_ps(tmp, tmp);

      // Extract the sum from the lowest element and add to thread-local
      // accumulator
      local_sum += _mm256_cvtss_f32(sum_vec);
    }

// Atomically add the thread-local sum to the global sum to avoid data races
#pragma omp atomic
    sum += local_sum;
  }

  // Phase 3: Normalize the output with masking
  // Compute reciprocal of sum (1/sum) once and broadcast to all vector lanes
  // for efficiency
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);

// Parallelize the normalization across available threads
#pragma omp parallel for num_threads(threads_to_use)
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    // Process data in cache-friendly blocks to minimize cache misses
    const size_t block_end = std::min(block_start + BLOCK_SIZE, K);

    size_t i = block_start;
    // Process 32 elements per iteration (4 AVX vectors = 32 floats) with loop
    // unrolling
    for (; i + 31 < block_end; i += 32) {
      // Load 4 AVX vectors (32 floats) from output buffer (containing
      // exp(x-max) values)
      __m256 data0 = _mm256_load_ps(output + i);
      __m256 data1 = _mm256_load_ps(output + i + 8);
      __m256 data2 = _mm256_load_ps(output + i + 16);
      __m256 data3 = _mm256_load_ps(output + i + 24);

      // Perform vectorized division by multiplying each vector by the inverse
      // sum (multiplication is faster than division in SIMD operations)
      data0 = _mm256_mul_ps(data0, inv_sum);
      data1 = _mm256_mul_ps(data1, inv_sum);
      data2 = _mm256_mul_ps(data2, inv_sum);
      data3 = _mm256_mul_ps(data3, inv_sum);

      // Store normalized results back to output buffer
      _mm256_store_ps(output + i, data0);
      _mm256_store_ps(output + i + 8, data1);
      _mm256_store_ps(output + i + 16, data2);
      _mm256_store_ps(output + i + 24, data3);
    }

    // Handle leftover elements in groups of 8 (one AVX vector)
    for (; i + 7 < block_end; i += 8) {
      __m256 data = _mm256_load_ps(output + i);
      data = _mm256_mul_ps(data, inv_sum); // Normalize with inverse sum
      _mm256_store_ps(output + i, data);   // Store result
    }

    // Handle remaining elements (less than 8) using masking for correct
    // boundary handling
    const size_t remaining = block_end - i;
    if (remaining > 0) {
      const __m256i mask =
          compute_mask(remaining); // Create mask for valid elements only
      __m256 data =
          _mm256_maskload_ps(output + i, mask); // Masked load of valid elements
      data = _mm256_mul_ps(data, inv_sum);      // Normalize with inverse sum
      _mm256_maskstore_ps(output + i, mask, data); // Masked store of results
    }
  }
}

/**
 * @brief AVX-accelerated softmax implementation for small input sizes.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 * @param num_threads Number of threads to use (default: -1, use all
 * available).
 *
 * This function is optimized for small arrays and avoids OpenMP overhead.
 */
void softmax_avx_small(const float *input, float *output, size_t K,
                       int num_threads = -1) {
  // The num_threads parameter is ignored in this implementation
  // since it's designed for small inputs where threading overhead exceeds
  // benefits

  __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  // Phase 1: Compute the maximum value using vectorized reduction with
  // masking Process 16 elements per iteration (2 AVX vectors) for better
  // throughput
  for (; i + 15 < K; i += 16) {
    const __m256 data1 = _mm256_load_ps(input + i);     // Load first 8 floats
    const __m256 data2 = _mm256_load_ps(input + i + 8); // Load next 8 floats
    max_vec = _mm256_max_ps(max_vec, data1); // Update maximum with first vector
    max_vec =
        _mm256_max_ps(max_vec, data2); // Update maximum with second vector
  }

  // Handle groups of 8 elements
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    max_vec = _mm256_max_ps(max_vec, data);
  }

  // Handle remaining elements (less than 8) using masking
  const size_t rem_phase1 = K - i;
  if (rem_phase1 > 0) {
    const __m256i mask =
        compute_mask(rem_phase1); // Create mask for valid elements only
    const __m256 data =
        _mm256_maskload_ps(input + i, mask); // Load only valid elements
    const __m256 blended = _mm256_blendv_ps(
        _mm256_set1_ps(-std::numeric_limits<float>::infinity()), data,
        _mm256_castsi256_ps(mask)); // Replace invalid elements with -infinity
    max_vec = _mm256_max_ps(max_vec, blended); // Update maximum
  }

  // Horizontal reduction to find the maximum value in the vector
  // Same pattern as the main implementation: cross-lane permutations followed
  // by in-lane shuffles
  __m256 tmp =
      _mm256_permute2f128_ps(max_vec, max_vec, 0x01); // Swap 128-bit lanes
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec,
                          _MM_SHUFFLE(1, 0, 3, 2)); // In-lane shuffle
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec,
                          _MM_SHUFFLE(2, 3, 0, 1)); // Final shuffle
  max_vec = _mm256_max_ps(max_vec, tmp);
  const float max_val =
      _mm256_cvtss_f32(max_vec); // Extract maximum value from first position

  // Phase 2: Compute exponentials and sum
  __m256 sum_vec = _mm256_setzero_ps(); // Initialize sum vector to zero
  const __m256 max_broadcast =
      _mm256_set1_ps(max_val); // Broadcast max value to all lanes
  i = 0;                       // Reset index counter

  // Process 16 elements per iteration (2 AVX vectors)
  for (; i + 15 < K; i += 16) {
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);
    // Subtract max_val for numerical stability before computing exponentials
    const __m256 exp1 = exp256_ps(_mm256_sub_ps(data1, max_broadcast));
    const __m256 exp2 = exp256_ps(_mm256_sub_ps(data2, max_broadcast));

    // Store intermediate exponential results
    _mm256_store_ps(output + i, exp1);
    _mm256_store_ps(output + i + 8, exp2);

    // Accumulate for sum calculation
    sum_vec = _mm256_add_ps(sum_vec, exp1);
    sum_vec = _mm256_add_ps(sum_vec, exp2);
  }

  // Process remaining groups of 8 elements
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
    _mm256_store_ps(output + i, exp);
    sum_vec = _mm256_add_ps(sum_vec, exp);
  }

  // Handle final elements (less than 8) with masking
  const size_t rem_phase2 = K - i;
  if (rem_phase2 > 0) {
    const __m256i mask =
        compute_mask(rem_phase2); // Create mask for valid elements
    const __m256 data = _mm256_maskload_ps(input + i, mask); // Masked load
    const __m256 exp =
        exp256_ps(_mm256_sub_ps(data, max_broadcast)); // Compute exponentials
    _mm256_maskstore_ps(output + i, mask,
                        exp); // Store only to valid positions

    // Zero out invalid lanes before accumulating to avoid corrupting the sum
    const __m256 blended =
        _mm256_blendv_ps(_mm256_setzero_ps(), exp, _mm256_castsi256_ps(mask));
    sum_vec = _mm256_add_ps(sum_vec, blended); // Add only valid elements to sum
  }

  // Efficient horizontal sum reduction using permute and hadd operations
  tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01); // Swap 128-bit lanes
  sum_vec = _mm256_add_ps(sum_vec, tmp);  // Add corresponding elements
  tmp = _mm256_hadd_ps(sum_vec, sum_vec); // Horizontal add within 128-bit lanes
  sum_vec =
      _mm256_hadd_ps(tmp, tmp); // Another horizontal add for final reduction
  float sum = _mm256_cvtss_f32(sum_vec); // Extract sum from first position

  // Phase 3: Normalize the output by dividing each element by the sum
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum); // Compute reciprocal once
  i = 0;                                             // Reset counter

  // Process 16 elements per iteration
  for (; i + 15 < K; i += 16) {
    __m256 data1 = _mm256_load_ps(output + i); // Load exponential results
    __m256 data2 = _mm256_load_ps(output + i + 8);

    // Multiply by 1/sum instead of dividing (faster SIMD operation)
    data1 = _mm256_mul_ps(data1, inv_sum);
    data2 = _mm256_mul_ps(data2, inv_sum);

    // Store normalized results
    _mm256_store_ps(output + i, data1);
    _mm256_store_ps(output + i + 8, data2);
  }

  // Process remaining groups of 8 elements
  for (; i + 7 < K; i += 8) {
    __m256 data = _mm256_load_ps(output + i);
    data = _mm256_mul_ps(data, inv_sum); // Normalize with inverse sum
    _mm256_store_ps(output + i, data);   // Store result
  }

  // Handle final elements with masking
  const size_t rem_phase3 = K - i;
  if (rem_phase3 > 0) {
    const __m256i mask = compute_mask(rem_phase3); // Mask for valid elements
    __m256 data =
        _mm256_maskload_ps(output + i, mask); // Load only valid elements
    data = _mm256_mul_ps(data, inv_sum);      // Normalize
    _mm256_maskstore_ps(output + i, mask,
                        data); // Store only to valid positions
  }
}

// --------------------------------------------------------------------------//
// This code implementation includes a standalone benchmarking mechanism with
// a main function that allows direct timing measurement of the softmax
// implementations. While you're supposed to use `make test`
// for formal benchmarking, this approach offers an alternative that directly
// prints the elapsed time using the TIMERSTART and TIMERSTOP macros from the
// original code.
// --------------------------------------------------------------------------//

/**
 * @brief Custom C++17 aligned memory allocator for AVX operations
 *
 * This allocator leverages C++17's aligned memory allocation features to
 * ensure that all allocated memory is properly aligned to 32-byte boundaries,
 * which is critical for optimal AVX vector operations that require aligned
 * memory access.
 *
 * The 32-byte alignment is chosen because:
 * - AVX/AVX2 registers are 256 bits (32 bytes) wide
 * - Aligned memory access is significantly faster than unaligned access
 * - Prevents potential crashes from unaligned memory access in strict
 * architectures
 *
 * @tparam T The type of elements to allocate
 */
template <typename T> class AlignedAllocatorC17 {
public:
  using value_type = T;
  static constexpr size_t alignment = 32; // Alignment required for AVX

  /**
   * @brief Allocate aligned memory for n elements of type T
   *
   * Uses C++17's aligned operator new to guarantee 32-byte alignment.
   *
   * @param n Number of elements to allocate
   * @return T* Pointer to aligned memory block
   */
  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t(alignment)));
  }

  /**
   * @brief Deallocate previously allocated memory
   *
   * Uses C++17's aligned operator delete to properly free memory.
   *
   * @param p Pointer to memory block
   * @param n Size of allocation (unused but required by allocator interface)
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
 * Provides a simple command-line interface for testing the softmax
 * implementations:
 * - First argument: Size of input array (K)
 * - Second argument: Optional flag to print results (any value works)
 *
 * The function automatically selects between softmax_avx and
 * softmax_avx_small based on the input size, with the threshold set at 2x
 * BLOCK_SIZE. This threshold was determined through empirical testing to
 * balance parallelization overhead vs. vectorization benefits.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return int Exit code (0 for success)
 */
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

  // Choose appropriate AVX function based on input size
  const size_t BLOCK_SIZE = 32 * 1024 / sizeof(float); // ~8192 floats

  // Benchmark AVX implementation
  if (K <= BLOCK_SIZE * 4) {
    std::printf("Using softmax_avx_small\n");
    TIMERSTART(softmax_avx_small);
    softmax_avx_small(input.data(), output.data(), K);
    TIMERSTOP(softmax_avx_small);
  } else {
    std::printf("Using softmax_avx\n");
    TIMERSTART(softmax_avx);
    softmax_avx(input.data(), output.data(), K);
    TIMERSTOP(softmax_avx);
  }

  // Print results if requested
  if (print) {
    printResult(output, K);
  }

  return 0;
}
#endif
