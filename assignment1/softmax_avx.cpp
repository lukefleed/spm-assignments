#include <algorithm>
#include <avx_mathfun.h>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static inline float horizontal_max(__m256 x) {
  __m256 y = _mm256_permute2f128_ps(x, x, 0x01); // Swap 128-bit lanes
  x = _mm256_max_ps(x, y);                       // Max tra le due metà
  x = _mm256_max_ps(x, _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2)));
  x = _mm256_max_ps(x, _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm256_cvtss_f32(x);
}

static inline float horizontal_sum(__m256 x) {
  __m256 y = _mm256_permute2f128_ps(x, x, 0x01);
  x = _mm256_add_ps(x, y);
  x = _mm256_hadd_ps(x, x);
  x = _mm256_hadd_ps(x, x);
  return _mm256_cvtss_f32(x);
}

void softmax_avx(const float *input, float *output, size_t K) {
  if (K == 0)
    return;

  // Step 1: Find maximum value
  __m256 max_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  // Process 64 elements (8 AVX registers) at once instead of 32
  for (; i + 64 <= K; i += 64) {
    _mm_prefetch(input + i + 64,
                 _MM_HINT_T0); // 64 elementi avanti (1 cache line)

    __m256 chunk1 = _mm256_loadu_ps(input + i);
    __m256 chunk2 = _mm256_loadu_ps(input + i + 8);
    __m256 chunk3 = _mm256_loadu_ps(input + i + 16);
    __m256 chunk4 = _mm256_loadu_ps(input + i + 24);
    __m256 chunk5 = _mm256_loadu_ps(input + i + 32);
    __m256 chunk6 = _mm256_loadu_ps(input + i + 40);
    __m256 chunk7 = _mm256_loadu_ps(input + i + 48);
    __m256 chunk8 = _mm256_loadu_ps(input + i + 56);

    max_vals = _mm256_max_ps(max_vals, chunk1);
    max_vals = _mm256_max_ps(max_vals, chunk2);
    max_vals = _mm256_max_ps(max_vals, chunk3);
    max_vals = _mm256_max_ps(max_vals, chunk4);
    max_vals = _mm256_max_ps(max_vals, chunk5);
    max_vals = _mm256_max_ps(max_vals, chunk6);
    max_vals = _mm256_max_ps(max_vals, chunk7);
    max_vals = _mm256_max_ps(max_vals, chunk8);
  }

  // Process remaining chunks of 32
  for (; i + 32 <= K; i += 32) {
    __m256 chunk1 = _mm256_loadu_ps(input + i);
    __m256 chunk2 = _mm256_loadu_ps(input + i + 8);
    __m256 chunk3 = _mm256_loadu_ps(input + i + 16);
    __m256 chunk4 = _mm256_loadu_ps(input + i + 24);

    max_vals = _mm256_max_ps(max_vals, chunk1);
    max_vals = _mm256_max_ps(max_vals, chunk2);
    max_vals = _mm256_max_ps(max_vals, chunk3);
    max_vals = _mm256_max_ps(max_vals, chunk4);
  }

  // Process remaining chunks of 8
  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(input + i);
    max_vals = _mm256_max_ps(max_vals, chunk);
  }

  float max_val = horizontal_max(max_vals);

  // Handle remainder elements
  for (; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // Step 2: Compute exponentials and sum
  __m256 sum_vec = _mm256_setzero_ps();
  __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0;

  // Process 32 elements at once
  for (; i + 32 <= K; i += 32) {
    // Prefetch ahead for both input and output arrays
    _mm_prefetch(input + i + 64, _MM_HINT_T0);
    _mm_prefetch(output + i + 64, _MM_HINT_T0);

    __m256 chunk1 = _mm256_loadu_ps(input + i);
    __m256 chunk2 = _mm256_loadu_ps(input + i + 8);
    __m256 chunk3 = _mm256_loadu_ps(input + i + 16);
    __m256 chunk4 = _mm256_loadu_ps(input + i + 24);

    chunk1 = _mm256_sub_ps(chunk1, max_broadcast);
    chunk2 = _mm256_sub_ps(chunk2, max_broadcast);
    chunk3 = _mm256_sub_ps(chunk3, max_broadcast);
    chunk4 = _mm256_sub_ps(chunk4, max_broadcast);

    __m256 exp_chunk1 = exp256_ps(chunk1);
    __m256 exp_chunk2 = exp256_ps(chunk2);
    __m256 exp_chunk3 = exp256_ps(chunk3);
    __m256 exp_chunk4 = exp256_ps(chunk4);

    _mm256_storeu_ps(output + i, exp_chunk1);
    _mm256_storeu_ps(output + i + 8, exp_chunk2);
    _mm256_storeu_ps(output + i + 16, exp_chunk3);
    _mm256_storeu_ps(output + i + 24, exp_chunk4);

    sum_vec = _mm256_add_ps(sum_vec, exp_chunk1);
    sum_vec = _mm256_add_ps(sum_vec, exp_chunk2);
    sum_vec = _mm256_add_ps(sum_vec, exp_chunk3);
    sum_vec = _mm256_add_ps(sum_vec, exp_chunk4);
  }

  // Process remaining chunks of 8
  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(input + i);
    chunk = _mm256_sub_ps(chunk, max_broadcast);
    __m256 exp_chunk = exp256_ps(chunk);
    _mm256_storeu_ps(output + i, exp_chunk);
    sum_vec = _mm256_add_ps(sum_vec, exp_chunk);
  }

  float sum = horizontal_sum(sum_vec);

  // Handle remainder elements
  for (; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  // Step 3: Normalize by sum - use multiplication by reciprocal instead of
  // division
  const float sum_recip = 1.0f / sum;
  __m256 sum_reciprocal = _mm256_set1_ps(sum_recip);
  i = 0;

  // Process 32 elements at once
  for (; i + 32 <= K; i += 32) {
    _mm_prefetch(output + i + 64, _MM_HINT_T0);

    __m256 chunk1 = _mm256_loadu_ps(output + i);
    __m256 chunk2 = _mm256_loadu_ps(output + i + 8);
    __m256 chunk3 = _mm256_loadu_ps(output + i + 16);
    __m256 chunk4 = _mm256_loadu_ps(output + i + 24);

    // Multiply by reciprocal (faster than division)
    chunk1 = _mm256_mul_ps(chunk1, sum_reciprocal);
    chunk2 = _mm256_mul_ps(chunk2, sum_reciprocal);
    chunk3 = _mm256_mul_ps(chunk3, sum_reciprocal);
    chunk4 = _mm256_mul_ps(chunk4, sum_reciprocal);

    _mm256_storeu_ps(output + i, chunk1);
    _mm256_storeu_ps(output + i + 8, chunk2);
    _mm256_storeu_ps(output + i + 16, chunk3);
    _mm256_storeu_ps(output + i + 24, chunk4);
  }

  // Process remaining chunks of 8
  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(output + i);
    chunk = _mm256_mul_ps(chunk, sum_reciprocal);
    _mm256_storeu_ps(output + i, chunk);
  }

  // Handle remainder elements
  for (; i < K; ++i) {
    output[i] *= sum_recip;
  }
}
