#include <algorithm>
#include <avx_mathfun.h>
#include <hpc_helpers.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static inline float horizontal_max(__m256 x) {
  __m128 vlow = _mm256_extractf128_ps(x, 0);
  __m128 vhigh = _mm256_extractf128_ps(x, 1);
  vlow = _mm_max_ps(vlow, vhigh);
  vhigh = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
  vlow = _mm_max_ps(vlow, vhigh);
  vhigh = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2));
  vlow = _mm_max_ps(vlow, vhigh);
  return _mm_cvtss_f32(vlow);
}

static inline float horizontal_sum(__m256 x) {
  __m128 a = _mm256_extractf128_ps(x, 0);
  __m128 b = _mm256_extractf128_ps(x, 1);
  a = _mm_add_ps(a, b);
  a = _mm_hadd_ps(a, a);
  a = _mm_hadd_ps(a, a);
  return _mm_cvtss_f32(a);
}

void softmax_avx(const float *input, float *output, size_t K) {
  if (K == 0)
    return;

  // Step 1: Find maximum value
  __m256 max_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  size_t i = 0;

  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(input + i);
    max_vals = _mm256_max_ps(max_vals, chunk);
  }

  float max_val = horizontal_max(max_vals);

  for (; i < K; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // Step 2: Compute exponentials and sum
  __m256 sum_vec = _mm256_setzero_ps();
  __m256 max_broadcast = _mm256_set1_ps(max_val);
  i = 0;

  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(input + i);
    chunk = _mm256_sub_ps(chunk, max_broadcast);
    __m256 exp_chunk = exp256_ps(chunk);
    _mm256_storeu_ps(output + i, exp_chunk);
    sum_vec = _mm256_add_ps(sum_vec, exp_chunk);
  }

  float sum = horizontal_sum(sum_vec);

  for (; i < K; ++i) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  // Step 3: Normalize by sum
  __m256 sum_broadcast = _mm256_set1_ps(sum);
  i = 0;

  for (; i + 8 <= K; i += 8) {
    __m256 chunk = _mm256_loadu_ps(output + i);
    chunk = _mm256_div_ps(chunk, sum_broadcast);
    _mm256_storeu_ps(output + i, chunk);
  }

  for (; i < K; ++i) {
    output[i] /= sum;
  }
}
