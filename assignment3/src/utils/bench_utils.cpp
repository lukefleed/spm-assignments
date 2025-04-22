#include "bench_utils.hpp"
#include <algorithm> // For sort, nth_element
#include <iostream>  // For potential error messages
#include <numeric>   // For accumulate
#include <vector>

namespace BenchUtils {

double run_benchmark(const std::function<bool()> &func_to_run, int iterations,
                     int warmup_iterations) {
  if (iterations < 1) {
    std::cerr << "Error: Benchmark requires at least 1 iteration." << std::endl;
    return -1.0;
  }
  if (warmup_iterations < 0) {
    std::cerr << "Error: Warmup iterations cannot be negative." << std::endl;
    return -1.0;
  }

  // --- Warmup Phase ---
  for (int i = 0; i < warmup_iterations; ++i) {
    if (!func_to_run()) {
      std::cerr << "Error: Function failed during warmup iteration " << i << "."
                << std::endl;
      return -2.0; // Indicate failure during warmup
    }
  }

  // --- Measurement Phase ---
  std::vector<double> timings_s;
  timings_s.reserve(iterations);
  BenchmarkTimer timer; // Reuse timer object

  for (int i = 0; i < iterations; ++i) {
    timer.reset(); // Start timer for this iteration

    if (!func_to_run()) {
      std::cerr << "Error: Function failed during measurement iteration " << i
                << "." << std::endl;
      return -3.0; // Indicate failure during measurement
    }

    timings_s.push_back(timer.elapsed_s()); // Record elapsed time
  }

  // --- Calculate Median ---
  if (timings_s.empty()) {
    // Should not happen if iterations >= 1, but check
    return -4.0;
  }

  // Use nth_element for efficiency in finding the median
  size_t mid_index = timings_s.size() / 2;
  std::nth_element(timings_s.begin(), timings_s.begin() + mid_index,
                   timings_s.end());

  if (timings_s.size() % 2 != 0) {
    // Odd number of elements, median is the middle one
    return timings_s[mid_index];
  } else {
    // Even number of elements, median is average of the two middle ones
    // nth_element puts the median-candidate at mid_index. We need the element
    // just before it too.
    double mid_val1 = timings_s[mid_index];
    // Find the maximum element in the lower half
    double mid_val0 =
        *std::max_element(timings_s.begin(), timings_s.begin() + mid_index);
    return (mid_val0 + mid_val1) / 2.0;
  }
}

} // namespace BenchUtils
