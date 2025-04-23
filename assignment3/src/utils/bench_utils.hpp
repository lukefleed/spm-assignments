#ifndef MINIZP_BENCH_UTILS_HPP
#define MINIZP_BENCH_UTILS_HPP

#include <chrono> // For high_resolution_clock
#include <functional>
#include <vector>

namespace BenchUtils {

/**
 * @brief Error codes for benchmarking results.
 */
enum class BenchmarkError {
  None = 0,
  InvalidIterations,
  InvalidWarmup,
  WarmupFailed,
  MeasurementFailed
};

/**
 * @brief Result of a benchmark run.
 */
struct BenchmarkResult {
  bool success;              /**< True if no error occurred. */
  double median_time_s;      /**< Median execution time in seconds. */
  BenchmarkError error_code; /**< Error code if success is false. */
};

/**
 * @brief Simple timer using high_resolution_clock.
 */
class BenchmarkTimer {
  std::chrono::high_resolution_clock::time_point start_time_;

public:
  BenchmarkTimer() : start_time_(std::chrono::high_resolution_clock::now()) {}
  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }
  double elapsed_s() const {
    auto diff = std::chrono::high_resolution_clock::now() - start_time_;
    return std::chrono::duration<double>(diff).count();
  }
  double elapsed_ms() const { return elapsed_s() * 1000.0; }
};

/**
 * @brief Runs a function multiple times and returns benchmark results.
 *
 * @param func_to_run The function to benchmark. Should return true on success.
 * @param iterations Number of measurement iterations. Must be >= 1.
 * @param warmup_iterations Number of warmup iterations. Must be >= 0.
 * @return BenchmarkResult Contains median time and error status.
 */
BenchmarkResult run_benchmark(const std::function<bool()> &func_to_run,
                              int iterations, int warmup_iterations);

} // namespace BenchUtils

#endif // MINIZP_BENCH_UTILS_HPP
