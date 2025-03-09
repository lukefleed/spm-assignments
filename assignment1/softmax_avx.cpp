#include <algorithm>
#include <avx_mathfun.h> // For AVX optimized exp
#include <immintrin.h>   // AVX intrinsics
#include <iostream>
#include <limits>
#include <omp.h> // For OpenMP parallelization
#include <random>
#include <vector>

// Improved AVX-accelerated softmax implementation
// with cache blocking, OpenMP, prefetching and optimized exp
void softmax_avx(const float *input, float *output, size_t K) {

  // Cache-friendly block size (L1 cache consideration)
  const size_t BLOCK_SIZE = 4096 / sizeof(float);

  // ========================================================================
  // Phase 1: Maximum value computation with vectorized reduction and blocking
  // ========================================================================
  float max_val = -std::numeric_limits<float>::infinity();

#pragma omp parallel
  {
    float local_max = -std::numeric_limits<float>::infinity();

#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

      // Prefetch next block
      if (block_start + BLOCK_SIZE < K) {
        _mm_prefetch((const char *)(input + block_start + BLOCK_SIZE),
                     _MM_HINT_T0);
      }

      size_t i = block_start;

      // Main processing loop with 4x unrolling (32 elements/iteration)
      for (; i + 31 < block_end; i += 32) {
        // Prefetch data ahead of time
        _mm_prefetch((const char *)(input + i + 64), _MM_HINT_T0);

        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

        max_vec = _mm256_max_ps(max_vec, data0);
        max_vec = _mm256_max_ps(max_vec, data1);
        max_vec = _mm256_max_ps(max_vec, data2);
        max_vec = _mm256_max_ps(max_vec, data3);
      }

      // Process remaining elements in 8-element chunks
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        max_vec = _mm256_max_ps(max_vec, data);
      }

      // Horizontal max reduction
      __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
      max_vec = _mm256_max_ps(max_vec, tmp);

      float block_max = _mm256_cvtss_f32(max_vec);

      // Handle remaining elements
      for (; i < block_end; ++i) {
        block_max = std::max(block_max, input[i]);
      }

      local_max = std::max(local_max, block_max);
    }

// Reduce local maximums to global maximum
#pragma omp critical
    { max_val = std::max(max_val, local_max); }
  }

  // ========================================================================
  // Phase 2: Exponential computation and sum reduction
  // ========================================================================
  float sum = 0.0f;

#pragma omp parallel
  {
    float local_sum = 0.0f;
    const __m256 max_broadcast = _mm256_set1_ps(max_val);

#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();

      // Prefetch next block
      if (block_start + BLOCK_SIZE < K) {
        _mm_prefetch((const char *)(input + block_start + BLOCK_SIZE),
                     _MM_HINT_T0);
      }

      size_t i = block_start;

      // Main processing loop with 4x unrolling
      for (; i + 31 < block_end; i += 32) {
        // Prefetch data ahead of time
        _mm_prefetch((const char *)(input + i + 64), _MM_HINT_T0);

        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

        const __m256 exp0 = exp256_ps(
            _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data0));
        const __m256 exp1 = exp256_ps(
            _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data1));
        const __m256 exp2 = exp256_ps(
            _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data2));
        const __m256 exp3 = exp256_ps(
            _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data3));

        _mm256_store_ps(output + i, exp0);
        _mm256_store_ps(output + i + 8, exp1);
        _mm256_store_ps(output + i + 16, exp2);
        _mm256_store_ps(output + i + 24, exp3);

        // Accumulate sums in parallel accumulators
        sum0 = _mm256_add_ps(sum0, _mm256_add_ps(exp0, exp1));
        sum1 = _mm256_add_ps(sum1, _mm256_add_ps(exp2, exp3));
      }

      // Process remaining elements in 8-element chunks
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_store_ps(output + i, exp);
        sum0 = _mm256_add_ps(sum0, exp);
      }

      // Handle remaining elements
      for (; i < block_end; ++i) {
        output[i] = expf(input[i] - max_val);
        local_sum += output[i];
      }

      // Usa masked processing
      if (i < block_end) {
        int remaining = block_end - i;
        float tmp[8] = {0}; // Allocazione temporanea allineata

        // Copia solo gli elementi rimanenti
        std::copy(input + i, input + block_end, tmp);

        const __m256 data = _mm256_load_ps(tmp);
        const __m256 exp_data = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_store_ps(tmp, exp_data);

        // Copia indietro solo gli elementi validi
        for (int j = 0; j < remaining; j++) {
          output[i + j] = tmp[j];
          local_sum += tmp[j];
        }
      }

      // Horizontal sum reduction
      __m256 sum_vec = _mm256_add_ps(sum0, sum1);
      __m256 tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
      sum_vec = _mm256_add_ps(sum_vec, tmp);
      tmp = _mm256_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm256_hadd_ps(tmp, tmp);

      local_sum += _mm256_cvtss_f32(sum_vec);
    }

// Reduce local sums to global sum
#pragma omp atomic
    sum += local_sum;
  }

  // ========================================================================
  // Phase 3: Normalization with vectorized division
  // ========================================================================
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);

#pragma omp parallel for
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
    size_t i = block_start;

    // Prefetch next block
    if (block_start + BLOCK_SIZE < K) {
      _mm_prefetch((const char *)(output + block_start + BLOCK_SIZE),
                   _MM_HINT_T0);
    }

    // Main processing loop with 4x unrolling
    for (; i + 31 < block_end; i += 32) {
      // Prefetch data ahead of time
      _mm_prefetch((const char *)(output + i + 64), _MM_HINT_T0);

      __m256 data0 = _mm256_load_ps(output + i);
      __m256 data1 = _mm256_load_ps(output + i + 8);
      __m256 data2 = _mm256_load_ps(output + i + 16);
      __m256 data3 = _mm256_load_ps(output + i + 24);

      data0 = _mm256_fmadd_ps(data0, inv_sum, _mm256_setzero_ps());
      data1 = _mm256_fmadd_ps(data1, inv_sum, _mm256_setzero_ps());
      data2 = _mm256_fmadd_ps(data2, inv_sum, _mm256_setzero_ps());
      data3 = _mm256_fmadd_ps(data3, inv_sum, _mm256_setzero_ps());

      _mm256_store_ps(output + i, data0);
      _mm256_store_ps(output + i + 8, data1);
      _mm256_store_ps(output + i + 16, data2);
      _mm256_store_ps(output + i + 24, data3);
    }

    // Process remaining elements in 8-element chunks
    for (; i + 7 < block_end; i += 8) {
      __m256 data = _mm256_load_ps(output + i);
      data = _mm256_mul_ps(data, inv_sum);
      _mm256_store_ps(output + i, data);
    }

    // Handle remaining elements
    for (; i < block_end; ++i) {
      output[i] /= sum;
    }
  }
}

/**
 * Optimized AVX implementation of softmax function for small input sizes
 * without OpenMP parallelization. This function computes:
 * output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
 *
 * @param input Pointer to input array (should be aligned to 32 bytes)
 * @param output Pointer to output array (should be aligned to 32 bytes)
 * @param K Size of input and output arrays
 */
void softmax_avx_small(const float *input, float *output, size_t K) {
  // ======================================================================
  // Phase 1: Find maximum value using vectorized operations
  // ======================================================================
  __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  // Process in 16-element chunks (2x unrolling) for better throughput
  for (; i + 15 < K; i += 16) {
    // Prefetch next chunk of data
    _mm_prefetch((const char *)(input + i + 16), _MM_HINT_T0);

    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);

    max_vec = _mm256_max_ps(max_vec, data1);
    max_vec = _mm256_max_ps(max_vec, data2);
  }

  // Process remaining elements in 8-element chunks
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    max_vec = _mm256_max_ps(max_vec, data);
  }

  // Horizontal max reduction - find maximum across all vector elements
  // First, compare high 128 bits with low 128 bits
  __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
  max_vec = _mm256_max_ps(max_vec, tmp);

  // Compare adjacent pairs of elements
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
  max_vec = _mm256_max_ps(max_vec, tmp);

  // Compare remaining adjacent pairs
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
  max_vec = _mm256_max_ps(max_vec, tmp);

  // Extract the maximum value (now in the lowest float of max_vec)
  float max_val = _mm256_cvtss_f32(max_vec);

  // Handle remaining elements sequentially
  for (; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // ======================================================================
  // Phase 2: Compute exponentials and sum
  // ======================================================================
  __m256 sum_vec1 = _mm256_setzero_ps(); // First accumulator
  __m256 sum_vec2 = _mm256_setzero_ps(); // Second accumulator for better ILP
  const __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0;

  // Process in 16-element chunks
  for (; i + 15 < K; i += 16) {
    // Prefetch next chunk
    _mm_prefetch((const char *)(input + i + 16), _MM_HINT_T0);

    // Load data and subtract max for numerical stability
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);
    const __m256 shifted1 =
        _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data1);
    const __m256 shifted2 =
        _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data2);

    // Compute exponentials
    const __m256 exp1 = exp256_ps(shifted1);
    const __m256 exp2 = exp256_ps(shifted2);

    // Store results
    _mm256_store_ps(output + i, exp1);
    _mm256_store_ps(output + i + 8, exp2);

    // Accumulate sums using two accumulators for better pipelining
    sum_vec1 = _mm256_add_ps(sum_vec1, exp1);
    sum_vec2 = _mm256_add_ps(sum_vec2, exp2);
  }

  // Process remaining elements in 8-element chunks
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    const __m256 shifted = _mm256_sub_ps(data, max_broadcast);
    const __m256 exp = exp256_ps(shifted);
    _mm256_store_ps(output + i, exp);
    sum_vec1 = _mm256_add_ps(sum_vec1, exp);
  }

  // Combine the two accumulators
  __m256 sum_vec = _mm256_add_ps(sum_vec1, sum_vec2);

  // Horizontal sum reduction - efficiently add all elements in the vector
  __m256 tmp_sum = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
  sum_vec = _mm256_add_ps(sum_vec, tmp_sum);
  tmp_sum = _mm256_hadd_ps(sum_vec, sum_vec);
  sum_vec = _mm256_hadd_ps(tmp_sum, tmp_sum);

  // Extract the sum (now in the lowest float)
  float sum = _mm256_cvtss_f32(sum_vec);

  // Handle remaining elements
  for (; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  // ======================================================================
  // Phase 3: Normalize by dividing by sum
  // ======================================================================
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
  i = 0;

  // Process in 16-element chunks
  for (; i + 15 < K; i += 16) {
    // Prefetch next chunk
    _mm_prefetch((const char *)(output + i + 16), _MM_HINT_T0);

    // Load, normalize and store
    __m256 data1 = _mm256_load_ps(output + i);
    __m256 data2 = _mm256_load_ps(output + i + 8);

    data1 = _mm256_mul_ps(data1, inv_sum);
    data2 = _mm256_mul_ps(data2, inv_sum);

    _mm256_store_ps(output + i, data1);
    _mm256_store_ps(output + i + 8, data2);
  }

  // use masked processing for remaining elements
  for (; i + 7 < K; i += 8) {
    __m256 data = _mm256_load_ps(output + i);
    data = _mm256_mul_ps(data, inv_sum);
    _mm256_store_ps(output + i, data);
  }
  // Handle remaining elements
  for (; i < K; ++i) {
    output[i] /= sum;
  }
}
