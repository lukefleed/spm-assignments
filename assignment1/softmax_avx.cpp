#include <algorithm>
#include <avx_mathfun.h>
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
 * @brief AVX-accelerated softmax implementation with masking and OpenMP
 * parallelization.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 *
 * This function computes the softmax of the input array using AVX instructions,
 * OpenMP parallelization, and masking to handle non-multiple-of-8 elements
 * efficiently. It is optimized for large arrays.
 */
void softmax_avx(const float *input, float *output, size_t K) {
  const size_t BLOCK_SIZE =
      32 * 1024 / sizeof(float); // Block size for cache-friendly processing
  float max_val = -std::numeric_limits<float>::infinity();

  // Phase 1: Compute the maximum value using vectorized reduction with masking
#pragma omp parallel num_threads(omp_get_num_procs())
  {
    float local_max = -std::numeric_limits<float>::infinity();

#pragma omp for nowait
    for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
      const size_t block_end = std::min(block_start + BLOCK_SIZE, K);
      __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

      size_t i = block_start;
      // Process 32 elements per iteration (4x unrolling)
      for (; i + 31 < block_end; i += 32) {
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

      // Process 8 elements per iteration
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        max_vec = _mm256_max_ps(max_vec, data);
      }

      // Handle remaining elements (less than 8) using masking
      const size_t remaining = block_end - i;
      if (remaining > 0) {
        const __m256i mask = compute_mask(remaining);
        const __m256 data = _mm256_maskload_ps(input + i, mask);
        const __m256 blended = _mm256_blendv_ps(
            _mm256_set1_ps(-std::numeric_limits<float>::infinity()), data,
            _mm256_castsi256_ps(mask));
        max_vec = _mm256_max_ps(max_vec, blended);
      }

      // Horizontal reduction to find the maximum value in the vector
      __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
      max_vec = _mm256_max_ps(max_vec, tmp);
      tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
      max_vec = _mm256_max_ps(max_vec, tmp);

      const float block_max = _mm256_cvtss_f32(max_vec);
      local_max = std::max(local_max, block_max);
    }

#pragma omp critical
    { max_val = std::max(max_val, local_max); }
  }

  // Phase 2: Compute exponentials and sum with masking
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

      size_t i = block_start;
      // Process 32 elements per iteration (4x unrolling)
      for (; i + 31 < block_end; i += 32) {
        const __m256 data0 = _mm256_load_ps(input + i);
        const __m256 data1 = _mm256_load_ps(input + i + 8);
        const __m256 data2 = _mm256_load_ps(input + i + 16);
        const __m256 data3 = _mm256_load_ps(input + i + 24);

        const __m256 exp0 = exp256_ps(_mm256_sub_ps(data0, max_broadcast));
        const __m256 exp1 = exp256_ps(_mm256_sub_ps(data1, max_broadcast));
        const __m256 exp2 = exp256_ps(_mm256_sub_ps(data2, max_broadcast));
        const __m256 exp3 = exp256_ps(_mm256_sub_ps(data3, max_broadcast));

        _mm256_store_ps(output + i, exp0);
        _mm256_store_ps(output + i + 8, exp1);
        _mm256_store_ps(output + i + 16, exp2);
        _mm256_store_ps(output + i + 24, exp3);

        sum0 = _mm256_add_ps(sum0, _mm256_add_ps(exp0, exp1));
        sum1 = _mm256_add_ps(sum1, _mm256_add_ps(exp2, exp3));
      }

      // Process 8 elements per iteration
      for (; i + 7 < block_end; i += 8) {
        const __m256 data = _mm256_load_ps(input + i);
        const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_store_ps(output + i, exp);
        sum0 = _mm256_add_ps(sum0, exp);
      }

      // Handle remaining elements (less than 8) using masking
      const size_t remaining = block_end - i;
      if (remaining > 0) {
        const __m256i mask = compute_mask(remaining);
        const __m256 data = _mm256_maskload_ps(input + i, mask);
        const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
        _mm256_maskstore_ps(output + i, mask, exp);

        const __m256 blended = _mm256_blendv_ps(_mm256_setzero_ps(), exp,
                                                _mm256_castsi256_ps(mask));
        sum0 = _mm256_add_ps(sum0, blended);
      }

      // Horizontal reduction to accumulate the sum
      __m256 sum_vec = _mm256_add_ps(sum0, sum1);
      __m256 tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
      sum_vec = _mm256_add_ps(sum_vec, tmp);
      tmp = _mm256_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm256_hadd_ps(tmp, tmp);
      local_sum += _mm256_cvtss_f32(sum_vec);
    }

#pragma omp atomic
    sum += local_sum;
  }

  // Phase 3: Normalize the output with masking
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
#pragma omp parallel for
  for (size_t block_start = 0; block_start < K; block_start += BLOCK_SIZE) {
    const size_t block_end = std::min(block_start + BLOCK_SIZE, K);

    size_t i = block_start;
    // Process 32 elements per iteration (4x unrolling)
    for (; i + 31 < block_end; i += 32) {
      __m256 data0 = _mm256_load_ps(output + i);
      __m256 data1 = _mm256_load_ps(output + i + 8);
      __m256 data2 = _mm256_load_ps(output + i + 16);
      __m256 data3 = _mm256_load_ps(output + i + 24);

      data0 = _mm256_mul_ps(data0, inv_sum);
      data1 = _mm256_mul_ps(data1, inv_sum);
      data2 = _mm256_mul_ps(data2, inv_sum);
      data3 = _mm256_mul_ps(data3, inv_sum);

      _mm256_store_ps(output + i, data0);
      _mm256_store_ps(output + i + 8, data1);
      _mm256_store_ps(output + i + 16, data2);
      _mm256_store_ps(output + i + 24, data3);
    }

    // Process 8 elements per iteration
    for (; i + 7 < block_end; i += 8) {
      __m256 data = _mm256_load_ps(output + i);
      data = _mm256_mul_ps(data, inv_sum);
      _mm256_store_ps(output + i, data);
    }

    // Handle remaining elements (less than 8) using masking
    const size_t remaining = block_end - i;
    if (remaining > 0) {
      const __m256i mask = compute_mask(remaining);
      __m256 data = _mm256_maskload_ps(output + i, mask);
      data = _mm256_mul_ps(data, inv_sum);
      _mm256_maskstore_ps(output + i, mask, data);
    }
  }
}

/**
 * @brief AVX-accelerated softmax implementation for small input sizes.
 * @param input Pointer to the input array (must be 32-byte aligned).
 * @param output Pointer to the output array (must be 32-byte aligned).
 * @param K Size of the input and output arrays.
 *
 * This function is optimized for small arrays and avoids OpenMP overhead.
 */
void softmax_avx_small(const float *input, float *output, size_t K) {
  __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  // Phase 1: Compute the maximum value using vectorized reduction with masking
  for (; i + 15 < K; i += 16) {
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);
    max_vec = _mm256_max_ps(max_vec, data1);
    max_vec = _mm256_max_ps(max_vec, data2);
  }

  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    max_vec = _mm256_max_ps(max_vec, data);
  }

  // Handle remaining elements (less than 8) using masking
  const size_t rem_phase1 = K - i;
  if (rem_phase1 > 0) {
    const __m256i mask = compute_mask(rem_phase1);
    const __m256 data = _mm256_maskload_ps(input + i, mask);
    const __m256 blended = _mm256_blendv_ps(
        _mm256_set1_ps(-std::numeric_limits<float>::infinity()), data,
        _mm256_castsi256_ps(mask));
    max_vec = _mm256_max_ps(max_vec, blended);
  }

  // Horizontal reduction to find the maximum value in the vector
  __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 0x01);
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
  max_vec = _mm256_max_ps(max_vec, tmp);
  tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
  max_vec = _mm256_max_ps(max_vec, tmp);
  const float max_val = _mm256_cvtss_f32(max_vec);

  // Phase 2: Compute exponentials and sum
  __m256 sum_vec = _mm256_setzero_ps();
  const __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0;

  for (; i + 15 < K; i += 16) {
    const __m256 data1 = _mm256_load_ps(input + i);
    const __m256 data2 = _mm256_load_ps(input + i + 8);
    const __m256 exp1 = exp256_ps(_mm256_sub_ps(data1, max_broadcast));
    const __m256 exp2 = exp256_ps(_mm256_sub_ps(data2, max_broadcast));

    _mm256_store_ps(output + i, exp1);
    _mm256_store_ps(output + i + 8, exp2);

    sum_vec = _mm256_add_ps(sum_vec, exp1);
    sum_vec = _mm256_add_ps(sum_vec, exp2);
  }

  for (; i + 7 < K; i += 8) {
    const __m256 data = _mm256_load_ps(input + i);
    const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
    _mm256_store_ps(output + i, exp);
    sum_vec = _mm256_add_ps(sum_vec, exp);
  }

  // Handle remaining elements (less than 8) using masking
  const size_t rem_phase2 = K - i;
  if (rem_phase2 > 0) {
    const __m256i mask = compute_mask(rem_phase2);
    const __m256 data = _mm256_maskload_ps(input + i, mask);
    const __m256 exp = exp256_ps(_mm256_sub_ps(data, max_broadcast));
    _mm256_maskstore_ps(output + i, mask, exp);

    const __m256 blended =
        _mm256_blendv_ps(_mm256_setzero_ps(), exp, _mm256_castsi256_ps(mask));
    sum_vec = _mm256_add_ps(sum_vec, blended);
  }

  // Sum reduction
  tmp = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
  sum_vec = _mm256_add_ps(sum_vec, tmp);
  tmp = _mm256_hadd_ps(sum_vec, sum_vec);
  sum_vec = _mm256_hadd_ps(tmp, tmp);
  float sum = _mm256_cvtss_f32(sum_vec);

  // Phase 3: Normalize the output with masking
  const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
  i = 0;

  for (; i + 15 < K; i += 16) {
    __m256 data1 = _mm256_load_ps(output + i);
    __m256 data2 = _mm256_load_ps(output + i + 8);

    data1 = _mm256_mul_ps(data1, inv_sum);
    data2 = _mm256_mul_ps(data2, inv_sum);

    _mm256_store_ps(output + i, data1);
    _mm256_store_ps(output + i + 8, data2);
  }

  for (; i + 7 < K; i += 8) {
    __m256 data = _mm256_load_ps(output + i);
    data = _mm256_mul_ps(data, inv_sum);
    _mm256_store_ps(output + i, data);
  }

  // Handle remaining elements (less than 8) using masking
  const size_t rem_phase3 = K - i;
  if (rem_phase3 > 0) {
    const __m256i mask = compute_mask(rem_phase3);
    __m256 data = _mm256_maskload_ps(output + i, mask);
    data = _mm256_mul_ps(data, inv_sum);
    _mm256_maskstore_ps(output + i, mask, data);
  }
}
