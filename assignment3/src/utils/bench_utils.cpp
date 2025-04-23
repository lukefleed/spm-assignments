/**
 * @file bench_utils.cpp
 * @brief Implementation of benchmarking utilities: warmup, measurement, and
 * median calculation.
 */

#include "bench_utils.hpp" // Declarations of benchmarking utilities
#include <algorithm>       // For nth_element, max_element
#include <iostream>        // For error messages
#include <vector>          // For timing data container

namespace BenchUtils {

/**
 * @brief Execute a function multiple times and compute the median execution
 * time.
 *
 * Performs a specified number of warmup runs, then measures execution time
 * over a number of iterations, and returns the median duration.
 *
 * @param func_to_run Callable returning true on success for each iteration.
 * @param iterations Number of measurement iterations (must be >= 1).
 * @param warmup_iterations Number of warmup iterations (must be >= 0).
 * @return BenchmarkResult containing median time and error code.
 */
BenchmarkResult run_benchmark(const std::function<bool()> &func_to_run,
                              int iterations, int warmup_iterations) {
  if (iterations < 1) {
    return {false, 0.0, BenchmarkError::InvalidIterations};
  }
  if (warmup_iterations < 0) {
    return {false, 0.0, BenchmarkError::InvalidWarmup};
  }

  // --- Warmup Phase: discard timing, check for early failures ---
  for (int i = 0; i < warmup_iterations; ++i) {
    if (!func_to_run()) {
      return {false, 0.0, BenchmarkError::WarmupFailed};
    }
  }

  // --- Measurement Phase: record execution times ---
  std::vector<double> timings_s;
  timings_s.reserve(iterations);
  BenchmarkTimer timer; // Timer for each iteration

  for (int i = 0; i < iterations; ++i) {
    timer.reset(); // Start timing

    if (!func_to_run()) {
      return {false, 0.0, BenchmarkError::MeasurementFailed};
    }

    timings_s.push_back(timer.elapsed_s()); // Store elapsed time
  }

  // --- Median Calculation: select middle value efficiently ---
  if (timings_s.empty()) {
    // Defensive check: should not occur when iterations >= 1
    return {false, 0.0, BenchmarkError::MeasurementFailed};
  }

  size_t mid_index = timings_s.size() / 2;
  std::nth_element(timings_s.begin(), timings_s.begin() + mid_index,
                   timings_s.end());

  if (timings_s.size() % 2 != 0) {
    // Odd count: exact middle is the median
    return {true, timings_s[mid_index], BenchmarkError::None};
  } else {
    // Even count: average two middle values
    double mid_hi = timings_s[mid_index];
    double mid_lo =
        *std::max_element(timings_s.begin(), timings_s.begin() + mid_index);
    return {true, (mid_lo + mid_hi) / 2.0, BenchmarkError::None};
  }
}

// Implementation of BenchmarkTimer methods
BenchmarkTimer::BenchmarkTimer()
    : start_time_(std::chrono::high_resolution_clock::now()) {}

void BenchmarkTimer::reset() {
  start_time_ = std::chrono::high_resolution_clock::now();
}

double BenchmarkTimer::elapsed_s() const {
  auto diff = std::chrono::high_resolution_clock::now() - start_time_;
  return std::chrono::duration<double>(diff).count();
}

double BenchmarkTimer::elapsed_ms() const { return elapsed_s() * 1000.0; }

} // namespace BenchUtils
