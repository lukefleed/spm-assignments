#include <algorithm>
#include <avx_mathfun.h>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

// Optimized AVX-accelerated softmax implementation with cache blocking,
// OpenMP parallelization, prefetching and loop unrolling.
void softmax_avx(const float *input, float *output, size_t K) {

  // Cache-friendly block size based on L1 cache size (4096 bytes assumed)
  // BLOCK_SIZE = number of floats that fit in 4096 bytes.
  const size_t BLOCK_SIZE = 32 * 1024 / sizeof(float); // 8192 floats

  // ------------------------------------------------------------------------
  // Phase 1: Compute the maximum value using vectorized reduction with
  // blocking.
  // ------------------------------------------------------------------------
  float max_val = -std::numeric_limits<float>::infinity();

#pragma omp parallel num_threads(omp_get_num_procs())
  {
    float local_max = -std::numeric_limits<float>::infinity();

#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

      // Prefetch the beginning of the next block to hide memory latency.
      if (block_start + BLOCK_SIZE < K) {
        _mm_prefetch(
            reinterpret_cast<const char *>(input + block_start + BLOCK_SIZE),
            _MM_HINT_T0);
      }

      size_t i = block_start;
      // Unroll loop 4x: process 32 elements per iteration.
      for (; i + 31 < block_end; i += 32) {
        // Prefetch data ahead in the block.
        _mm_prefetch(reinterpret_cast<const char *>(input + i + 128),
                     _MM_HINT_T0);

        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

        max_vec = _mm256_max_ps(max_vec, data0);
        max_vec = _mm256_max_ps(max_vec, data1);
        max_vec = _mm256_max_ps(max_vec, data2);
        max_vec = _mm256_max_ps(max_vec, data3);
      }

      // Process remaining elements in 8-element chunks.
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        max_vec = _mm256_max_ps(max_vec, data);
      }

      // Horizontal reduction of max_vec:
      __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
      max_vec = _mm256_max_ps(max_vec, tmp);

      float block_max = _mm256_cvtss_f32(max_vec);

      // Process any remaining elements scalar-wise.
      for (; i < block_end; ++i) {
        block_max = std::max(block_max, input[i]);
      }

      local_max = std::max(local_max, block_max);
    }

    // Merge local maximum values into the global maximum.
#pragma omp critical
    { max_val = std::max(max_val, local_max); }
  }

  // ------------------------------------------------------------------------
  // Phase 2: Compute exponentials and reduce the sum.
  // ------------------------------------------------------------------------
  float sum = 0.0f;
#pragma omp parallel
  {
    float local_sum = 0.0f;
    const __m256 max_broadcast = _mm256_set1_ps(max_val);
    // Precompute constant 1.0 for use with _mm256_fnmadd_ps.
    const __m256 one = _mm256_set1_ps(1.0f);

#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();

      // Prefetch next block from input.
      if (block_start + BLOCK_SIZE < K) {
        _mm_prefetch(
            reinterpret_cast<const char *>(input + block_start + BLOCK_SIZE),
            _MM_HINT_T0);
      }

      size_t i = block_start;
      // Unroll loop 4x: process 32 elements per iteration.
      for (; i + 31 < block_end; i += 32) {
        _mm_prefetch(reinterpret_cast<const char *>(input + i + 128),
                     _MM_HINT_T0);

        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

        // Compute (data - max_val) using fnmadd: -(1.0 * max) + data.
        const __m256 exp0 =
            exp256_ps(_mm256_fnmadd_ps(one, max_broadcast, data0));
        const __m256 exp1 =
            exp256_ps(_mm256_fnmadd_ps(one, max_broadcast, data1));
        const __m256 exp2 =
            exp256_ps(_mm256_fnmadd_ps(one, max_broadcast, data2));
        const __m256 exp3 =
            exp256_ps(_mm256_fnmadd_ps(one, max_broadcast, data3));

        _mm256_store_ps(output + i, exp0);
        _mm256_store_ps(output + i + 8, exp1);
        _mm256_store_ps(output + i + 16, exp2);
        _mm256_store_ps(output + i + 24, exp3);

        // Accumulate the exponentials in two accumulators.
        sum0 = _mm256_add_ps(sum0, _mm256_add_ps(exp0, exp1));
        sum1 = _mm256_add_ps(sum1, _mm256_add_ps(exp2, exp3));
      }

      // Process remaining elements in 8-element chunks.
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_store_ps(output + i, exp);
        sum0 = _mm256_add_ps(sum0, exp);
      }

      // Handle any remaining elements scalar-wise.
      for (; i < block_end; ++i) {
        output[i] = expf(input[i] - max_val);
        local_sum += output[i];
      }

      // Horizontal reduction of the two accumulators.
      __m256 sum_vec = _mm256_add_ps(sum0, sum1);
      __m256 tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
      sum_vec = _mm256_add_ps(sum_vec, tmp);
      tmp = _mm256_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm256_hadd_ps(tmp, tmp);
      local_sum += _mm256_cvtss_f32(sum_vec);
    }

    // Atomically add each thread's local sum to the global sum.
#pragma omp atomic
    sum += local_sum;
  }

  // ------------------------------------------------------------------------
  // Phase 3: Normalize the output with vectorized division.
  // ------------------------------------------------------------------------
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);

#pragma omp parallel for
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
    size_t i = block_start;

    // Prefetch the next block from output.
    if (block_start + BLOCK_SIZE < K) {
      _mm_prefetch(
          reinterpret_cast<const char *>(output + block_start + BLOCK_SIZE),
          _MM_HINT_T0);
    }

    // Unroll loop 4x: process 32 elements per iteration.
    for (; i + 31 < block_end; i += 32) {
      _mm_prefetch(reinterpret_cast<const char *>(output + i + 128),
                   _MM_HINT_T0);

      __m256 data0 = _mm256_load_ps(output + i);
      __m256 data1 = _mm256_load_ps(output + i + 8);
      __m256 data2 = _mm256_load_ps(output + i + 16);
      __m256 data3 = _mm256_load_ps(output + i + 24);

      // Use plain multiplication for clarity; equivalent to a fused
      // multiply-add with zero.
      data0 = _mm256_mul_ps(data0, inv_sum);
      data1 = _mm256_mul_ps(data1, inv_sum);
      data2 = _mm256_mul_ps(data2, inv_sum);
      data3 = _mm256_mul_ps(data3, inv_sum);

      _mm256_store_ps(output + i, data0);
      _mm256_store_ps(output + i + 8, data1);
      _mm256_store_ps(output + i + 16, data2);
      _mm256_store_ps(output + i + 24, data3);
    }

    // Process remaining elements in 8-element chunks.
    for (; i + 7 < block_end; i += 8) {
      __m256 data = _mm256_load_ps(output + i);
      data = _mm256_mul_ps(data, inv_sum);
      _mm256_store_ps(output + i, data);
    }

    // Handle any final elements scalar-wise.
    for (; i < block_end; ++i) {
      output[i] /= sum;
    }
  }
}

/**
 * Optimized AVX implementation of the softmax function for small input sizes.
 * This version does not use OpenMP and is tuned for lower overhead on small
 * arrays. Computes: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
 *
 * @param input  Pointer to input array (should be 32-byte aligned)
 * @param output Pointer to output array (should be 32-byte aligned)
 * @param K      Size of the input and output arrays
 */
void softmax_avx_small(const float *input, float *output, size_t K) {
  // ------------------------------------------------------------------------
  // Phase 1: Compute maximum value using vectorized operations.
  // ------------------------------------------------------------------------
  __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  // Process in 16-element chunks (2x unrolling).
  for (; i + 15 < K; i += 16) {
    _mm_prefetch(reinterpret_cast<const char *>(input + i + 128), _MM_HINT_T0);
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);
    max_vec = _mm256_max_ps(max_vec, data1);
    max_vec = _mm256_max_ps(max_vec, data2);
  }

  // Process remaining elements in 8-element chunks.
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    max_vec = _mm256_max_ps(max_vec, data);
  }

  // Horizontal reduction to get the maximum value.
  __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
  max_vec = _mm256_max_ps(max_vec, tmp);
  float max_val = _mm256_cvtss_f32(max_vec);

  // Process any remaining elements scalar-wise.
  for (; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // ------------------------------------------------------------------------
  // Phase 2: Compute exponentials and accumulate the sum.
  // ------------------------------------------------------------------------
  __m256 sum_vec1 = _mm256_setzero_ps(); // Accumulator 1
  __m256 sum_vec2 = _mm256_setzero_ps(); // Accumulator 2 for increased
                                         // instruction-level parallelism
  const __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0;

  // Process in 16-element chunks.
  for (; i + 15 < K; i += 16) {
    _mm_prefetch(reinterpret_cast<const char *>(input + i + 128), _MM_HINT_T0);

    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);

    // Using fnmadd to compute (data - max_val)
    const __m256 shifted1 =
        _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data1);
    const __m256 shifted2 =
        _mm256_fnmadd_ps(_mm256_set1_ps(1.0f), max_broadcast, data2);

    const __m256 exp1 = exp256_ps(shifted1);
    const __m256 exp2 = exp256_ps(shifted2);

    _mm256_store_ps(output + i, exp1);
    _mm256_store_ps(output + i + 8, exp2);

    sum_vec1 = _mm256_add_ps(sum_vec1, exp1);
    sum_vec2 = _mm256_add_ps(sum_vec2, exp2);
  }

  // Process remaining elements in 8-element chunks.
  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    const __m256 shifted = _mm256_sub_ps(data, max_broadcast);
    const __m256 exp = exp256_ps(shifted);
    _mm256_store_ps(output + i, exp);
    sum_vec1 = _mm256_add_ps(sum_vec1, exp);
  }

  // Combine the two accumulators.
  __m256 sum_vec = _mm256_add_ps(sum_vec1, sum_vec2);
  __m256 tmp_sum = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
  sum_vec = _mm256_add_ps(sum_vec, tmp_sum);
  tmp_sum = _mm256_hadd_ps(sum_vec, sum_vec);
  sum_vec = _mm256_hadd_ps(tmp_sum, tmp_sum);

  // Extract the sum from the vector.
  float sum = _mm256_cvtss_f32(sum_vec);

  // Process any remaining elements scalar-wise.
  for (; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  // ------------------------------------------------------------------------
  // Phase 3: Normalize by dividing each exponential by the sum.
  // ------------------------------------------------------------------------
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
  i = 0;

  // Process in 16-element chunks.
  for (; i + 15 < K; i += 16) {
    _mm_prefetch(reinterpret_cast<const char *>(output + i + 128), _MM_HINT_T0);

    __m256 data1 = _mm256_load_ps(output + i);
    __m256 data2 = _mm256_load_ps(output + i + 8);

    data1 = _mm256_mul_ps(data1, inv_sum);
    data2 = _mm256_mul_ps(data2, inv_sum);

    _mm256_store_ps(output + i, data1);
    _mm256_store_ps(output + i + 8, data2);
  }

  // Process remaining elements in 8-element chunks.
  for (; i + 7 < K; i += 8) {
    __m256 data = _mm256_load_ps(output + i);
    data = _mm256_mul_ps(data, inv_sum);
    _mm256_store_ps(output + i, data);
  }

  // Handle final elements scalar-wise.
  for (; i < K; ++i) {
    output[i] /= sum;
  }
}
