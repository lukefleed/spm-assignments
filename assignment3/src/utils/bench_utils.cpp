#include "bench_utils.hpp"
#include <algorithm> // For sort, nth_element
#include <iostream>  // For potential error messages
#include <numeric>   // For accumulate
#include <vector>

namespace BenchUtils {

BenchmarkResult run_benchmark(const std::function<bool()> &func_to_run,
                              int iterations, int warmup_iterations) {
  if (iterations < 1) {
    return {false, 0.0, BenchmarkError::InvalidIterations};
  }
  if (warmup_iterations < 0) {
    return {false, 0.0, BenchmarkError::InvalidWarmup};
  }

  // --- Warmup Phase ---
  for (int i = 0; i < warmup_iterations; ++i) {
    if (!func_to_run()) {
      return {false, 0.0, BenchmarkError::WarmupFailed};
    }
  }

  // --- Measurement Phase ---
  std::vector<double> timings_s;
  timings_s.reserve(iterations);
  BenchmarkTimer timer; // Reuse timer object

  for (int i = 0; i < iterations; ++i) {
    timer.reset(); // Start timer for this iteration

    if (!func_to_run()) {
      return {false, 0.0, BenchmarkError::MeasurementFailed};
    }

    timings_s.push_back(timer.elapsed_s()); // Record elapsed time
  }

  // --- Calculate Median ---
  if (timings_s.empty()) {
    // Should not happen if iterations >= 1, but defensive
    return {false, 0.0, BenchmarkError::MeasurementFailed};
  }

  // Use nth_element for efficiency in finding the median
  size_t mid_index = timings_s.size() / 2;
  std::nth_element(timings_s.begin(), timings_s.begin() + mid_index,
                   timings_s.end());

  if (timings_s.size() % 2 != 0) {
    // Odd number of elements, median is the middle one
    return {true, timings_s[mid_index], BenchmarkError::None};
  } else {
    // Even number of elements, median is average of the two middle ones
    // nth_element puts the median-candidate at mid_index. We need the element
    // just before it too.
    double mid_val1 = timings_s[mid_index];
    // Find the maximum element in the lower half
    double mid_val0 =
        *std::max_element(timings_s.begin(), timings_s.begin() + mid_index);
    return {true, (mid_val0 + mid_val1) / 2.0, BenchmarkError::None};
  }
}

} // namespace BenchUtils
