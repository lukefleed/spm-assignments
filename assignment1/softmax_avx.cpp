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
 * @brief Computes the softmax function using AVX vectorization and OpenMP
 * parallelization.
 *
 * This function implements a optimized softmax computation with three phases:
 * 1. Find the maximum value across all elements for numerical stability
 * 2. Compute exponentials and their sum using exp(x - max_val)
 * 3. Normalize by dividing each exponential by the sum
 *
 * @param input Pointer to input array of floats (must be 32-byte aligned for
 * AVX)
 * @param output Pointer to output array of floats (must be 32-byte aligned for
 * AVX)
 * @param K Number of elements in the input/output arrays
 * @param num_threads Number of OpenMP threads to use (-1 for auto-detection)
 *
 * @note The input and output arrays must be properly aligned for AVX operations
 */
void softmax_avx(const float *input, float *output, size_t K,
                 int num_threads = -1) {
  // Block size for cache-friendly processing (approximately 8K floats)
  const size_t BLOCK_SIZE = 32 * 1024 / sizeof(float);
  // Initialize overall maximum to the most negative finite value
  float max_val = -std::numeric_limits<float>::max();

  // Use specified thread count or default to processor count
  int threads_to_use = (num_threads > 0) ? num_threads : omp_get_num_procs();
  // PHASE 1: Compute the maximum value across all elements
  // The reduction clause allows each thread to compute its own maximum, which
  // is then combined at the end.
#pragma omp parallel for reduction(max : max_val) num_threads(threads_to_use)
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    const size_t block_end =
        std::min(block_start + BLOCK_SIZE,
                 K); // Handle last block potentially being smaller
    // Initialize max_vec to the most negative finite value for each block
    // This creates 8 copies of the most negative finite value in the 256-bit
    // register
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::max());

    size_t i = block_start;
    // Loop unrolling and prefetching
    // Process 32 elements per iteration (4x unrolling of AVX 8-float vectors)
    for (; i + 31 < block_end; i += 32) {
      // Prefetch 64 bytes (a cache line) ahead to improve cache hit rate from
      // the specified input position. The data will be available in the L1
      // cache of _MM_HINT_T0
      _mm_prefetch(reinterpret_cast<const char *>(input + i + 128),
                   _MM_HINT_T0);

      // Load 4 AVX vectors (32 floats) from the input array
      const __m256 data0 = _mm256_load_ps(input + i);
      const __m256 data1 = _mm256_load_ps(input + i + 8);
      const __m256 data2 = _mm256_load_ps(input + i + 16);
      const __m256 data3 = _mm256_load_ps(input + i + 24);

      // Update max_vec by comparing with each data vector
      max_vec = _mm256_max_ps(max_vec, data0);
      max_vec = _mm256_max_ps(max_vec, data1);
      max_vec = _mm256_max_ps(max_vec, data2);
      max_vec = _mm256_max_ps(max_vec, data3);
    }

    // Handle leftover elements in groups of 8 (one AVX vector)
    for (; i + 7 < block_end; i += 8) {
      const __m256 data = _mm256_load_ps(input + i);
      max_vec = _mm256_max_ps(max_vec, data);
    }

    // Handle remaining elements (less than 8) using masking
    const size_t remaining = block_end - i;
    if (remaining > 0) {
      // Create mask where only valid elements are active
      const __m256i mask = compute_mask(remaining);
      // Masked load of remaining elements
      // This will load only the first `remaining` elements and zero out the
      // rest
      const __m256 data = _mm256_maskload_ps(input + i, mask);
      // Since masked zeros all invalid lanes, and 0 may be a valid max, we need
      // to change (blend) this zeros with the most negative finite value.
      // `blendv_ps` blends two vectors based on a mask, replacing invalid lanes
      // with the most negative finite value. `castsi256_ps` reinterprets the
      // bits of a register as a float vector, without changing the underlying
      // data.
      const __m256 blended =
          _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::max()),
                           data, _mm256_castsi256_ps(mask));
      // Update max_vec with the masked data
      max_vec = _mm256_max_ps(max_vec, blended);
    }

    // At this points, max_vec contains the 8 partial maximums. To collapse them
    // in one scalar, we use a sequence of shuffles (a horizontal reduction).

    // Exchange the high and low 128-bit lanes of the max_vec vector
    // This is done to prepare for a horizontal reduction across all lanes
    // of the vector. The 0x01 indicates that we want to swap the high
    // and low 128-bit lanes.
    __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);

    // Compare and update max_vec with the swapped version
    max_vec = _mm256_max_ps(max_vec, tmp);

    // Then shuffle within 128-bit lanes and compare
    // `_MM_SHUFFLE(1, 0, 3, 2)` re-orders the elements every half (4 floats)
    // This makes closer elements to be compared
    tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));

    // Compare and update max_vec with the shuffled version
    max_vec = _mm256_max_ps(max_vec, tmp);

    // Final shuffle and max to get the maximum in all positions
    tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));

    // Compare and update max_vec with the final shuffled version
    max_vec = _mm256_max_ps(max_vec, tmp);

    // Extract the maximum value from lane 0 of the max_vec vector register
    // `cvtss` means "convert scalar single-precision"
    const float block_max = _mm256_cvtss_f32(max_vec);
    max_val = std::max(max_val, block_max); // Update global maximum
    // Alternative using intrinsic: max_val = fmaxf(max_val, block_max);
  }

  // Phase 2: Compute exponentials and sum
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
        // Prefetch 64 bytes (a cache line) ahead to improve cache hit rate
        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

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
        sum0 = _mm256_add_ps(sum0, _mm256_add_ps(exp0, exp1));
        sum1 = _mm256_add_ps(sum1, _mm256_add_ps(exp2, exp3));
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
      __m256 tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
      sum_vec = _mm256_add_ps(sum_vec, tmp);

      // Step 2: Horizontal sum of two registers. Computes in pairs (a0+a1,
      // a2+a3, b0+b1, b2+b3) and then puts the results of both source registers
      // in the destination register.
      tmp = _mm256_hadd_ps(sum_vec, sum_vec);

      // Step 3: Another horizontal add to get the final sum in the lowest
      // element
      sum_vec = _mm256_hadd_ps(tmp, tmp);

      // Extract the sum from the lowest element (lane 0) and add to
      // thread-local accumulator
      local_sum += _mm256_cvtss_f32(sum_vec);
    }

// Atomically add the thread-local sum to the global sum to avoid data races
#pragma omp atomic
    sum += local_sum;
  }

  // Phase 3: Normalize the output with masking
  // Compute reciprocal of sum (1/sum) once and broadcast to all vector lanes
  // for efficiency. The broadcast is performed by `_mm256_set1_ps` itself
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
 * @brief AVX-accelerated softmax implementation optimized for small input
 * sizes.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 * @param num_threads Number of threads to use. This parameter is ignored.
 *
 * This function is designed for small arrays where OpenMP threading overhead
 * would be detrimental. It implements the same AVX logic as the parallel
 * version but in a purely sequential context.
 */
void softmax_avx_small(const float *input, float *output, size_t K,
                       int num_threads = -1) {
  // The num_threads parameter is ignored in this implementation, as it is
  // designed for small inputs where threading overhead exceeds benefits.

  // Initialize max_vec to the most negative finite value for the reduction.
  // This creates 8 copies of the most negative finite value in the 256-bit
  // register.
  __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::max());
  size_t i = 0;

  // Phase 1: Compute the maximum value using a vectorized reduction with
  // masking. The main loop is unrolled to process 16 elements (2 AVX vectors)
  // per iteration to improve instruction-level parallelism and throughput.
  for (; i + 15 < K; i += 16) {
    // Load two full AVX vectors (16 floats) from the input array.
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);

    // Perform a packed maximum operation on each vector, updating the
    // accumulator.
    max_vec = _mm256_max_ps(max_vec, data1);
    max_vec = _mm256_max_ps(max_vec, data2);
  }

  // Handle any remaining full groups of 8 elements.
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    max_vec = _mm256_max_ps(max_vec, data);
  }

  // Handle the final remaining elements (less than 8) using a masked load.
  const size_t rem_phase1 = K - i;
  if (rem_phase1 > 0) {
    // Create a mask to activate only the lanes for valid remaining elements.
    const __m256i mask = compute_mask(rem_phase1);

    // Load only the valid elements; invalid lanes in the destination register
    // are zeroed.
    const __m256 data = _mm256_maskload_ps(input + i, mask);

    // Since masked lanes are zeroed, and 0 might be a valid maximum, we must
    // replace these zeros with the most negative value before comparison.
    // `blendv_ps` selects from two source vectors based on the mask.
    const __m256 blended =
        _mm256_blendv_ps(_mm256_set1_ps(-std::numeric_limits<float>::max()),
                         data, _mm256_castsi256_ps(mask));

    // Update the maximum with the safely blended data.
    max_vec = _mm256_max_ps(max_vec, blended);
  }

  // At this point, max_vec contains 8 partial maximums. To collapse them
  // into one scalar, we use a sequence of shuffles (a horizontal reduction).
  __m256 tmp;

  // 1. Exchange the high and low 128-bit lanes of the max_vec vector.
  //    This brings elements from opposite ends of the register next to each
  //    other for comparison. The 0x01 is the immediate control for the
  //    permutation.
  tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
  max_vec = _mm256_max_ps(max_vec, tmp);

  // 2. Shuffle elements within each 128-bit lane.
  //    _MM_SHUFFLE(1, 0, 3, 2) re-orders elements to continue the reduction.
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
  max_vec = _mm256_max_ps(max_vec, tmp);

  // 3. Final shuffle to ensure the maximum value is in all lanes.
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
  max_vec = _mm256_max_ps(max_vec, tmp);

  // 4. Extract the final scalar maximum value from lane 0.
  const float max_val = _mm256_cvtss_f32(max_vec);

  // Phase 2: Compute exponentials and their sum.
  // Initialize the sum accumulator vector to all zeros.
  __m256 sum_vec = _mm256_setzero_ps();
  // Broadcast the scalar max_val to all 8 lanes for efficient subtraction.
  const __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0; // Reset index for the new pass.

  // Process 16 elements per iteration (2 AVX vectors) for better throughput.
  for (; i + 15 < K; i += 16) {
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);

    // Compute exp(x - max_val) for numerical stability.
    const __m256 exp1 = exp256_ps(_mm256_sub_ps(data1, max_broadcast));
    const __m256 exp2 = exp256_ps(_mm256_sub_ps(data2, max_broadcast));

    // Store the intermediate exponential results to the output array.
    _mm256_store_ps(output + i, exp1);
    _mm256_store_ps(output + i + 8, exp2);

    // Accumulate the results for the sum calculation.
    sum_vec = _mm256_add_ps(sum_vec, exp1);
    sum_vec = _mm256_add_ps(sum_vec, exp2);
  }

  // Process any remaining full groups of 8 elements.
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
    _mm256_store_ps(output + i, exp);
    sum_vec = _mm256_add_ps(sum_vec, exp);
  }

  // Handle the final remaining elements (less than 8) using masking.
  const size_t rem_phase2 = K - i;
  if (rem_phase2 > 0) {
    const __m256i mask = compute_mask(rem_phase2);
    const __m256 data = _mm256_maskload_ps(input + i, mask);
    const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));

    // Store results only to valid memory locations.
    _mm256_maskstore_ps(output + i, mask, exp);

    // To avoid corrupting the sum, blend the exponential results with zero.
    // Only the valid, active lanes will contribute to the sum.
    const __m256 blended =
        _mm256_blendv_ps(_mm256_setzero_ps(), exp, _mm256_castsi256_ps(mask));
    sum_vec = _mm256_add_ps(sum_vec, blended);
  }

  // Perform an efficient horizontal sum reduction on the `sum_vec` accumulator.
  // 1. Swap and add the 128-bit lanes.
  tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
  sum_vec = _mm256_add_ps(sum_vec, tmp);
  // 2. Use horizontal add twice to fully reduce the vector.
  tmp = _mm256_hadd_ps(sum_vec, sum_vec);
  sum_vec = _mm256_hadd_ps(tmp, tmp);
  // 3. Extract the final scalar sum from lane 0.
  float sum = _mm256_cvtss_f32(sum_vec);

  // Phase 3: Normalize the output by dividing each element by the sum.
  // Pre-calculate the reciprocal of the sum to replace slow divisions with fast
  // multiplications.
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
  i = 0; // Reset index for the final pass.

  // Process 16 elements per iteration.
  for (; i + 15 < K; i += 16) {
    // Load the intermediate exponential results from the output buffer.
    __m256 data1 = _mm256_load_ps(output + i);
    __m256 data2 = _mm256_load_ps(output + i + 8);

    // Normalize by multiplying with the inverse sum.
    data1 = _mm256_mul_ps(data1, inv_sum);
    data2 = _mm256_mul_ps(data2, inv_sum);

    // Store the final normalized results back to the output buffer.
    _mm256_store_ps(output + i, data1);
    _mm256_store_ps(output + i + 8, data2);
  }

  // Process any remaining full groups of 8 elements.
  for (; i + 7 < K; i += 8) {
    __m256 data = _mm256_load_ps(output + i);
    data = _mm256_mul_ps(data, inv_sum);
    _mm256_store_ps(output + i, data);
  }

  // Handle the final remaining elements (less than 8) with a masked store.
  const size_t rem_phase3 = K - i;
  if (rem_phase3 > 0) {
    const __m256i mask = compute_mask(rem_phase3);
    // Load only valid elements from memory.
    __m256 data = _mm256_maskload_ps(output + i, mask);
    // Normalize.
    data = _mm256_mul_ps(data, inv_sum);
    // Store results only to valid memory locations, avoiding buffer overruns.
    _mm256_maskstore_ps(output + i, mask, data);
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
