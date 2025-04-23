/**
 * @file bench_utils.hpp
 * @brief Declarations of benchmarking utilities: error codes, result struct,
 * high-resolution timer, and benchmark runner.
 */
#ifndef MINIZP_BENCH_UTILS_HPP
#define MINIZP_BENCH_UTILS_HPP

#include <chrono> // For high_resolution_clock
#include <functional>

namespace BenchUtils {

/**
 * @enum BenchmarkError
 * @brief Error codes representing possible failures during benchmarking.
 */
enum class BenchmarkError {
  None = 0,          /**< No error. */
  InvalidIterations, /**< iterations < 1. */
  InvalidWarmup,     /**< warmup_iterations < 0. */
  WarmupFailed,      /**< A warmup iteration returned false. */
  MeasurementFailed  /**< A measurement iteration returned false or timings
                        empty. */
};

/**
 * @struct BenchmarkResult
 * @brief Stores the outcome of a benchmark run.
 *
 * @var BenchmarkResult::success
 * True if no error occurred.
 * @var BenchmarkResult::median_time_s
 * Median execution time of measurement iterations in seconds.
 * @var BenchmarkResult::error_code
 * Error code if success is false; otherwise BenchmarkError::None.
 */
struct BenchmarkResult {
  bool success;
  double median_time_s;
  BenchmarkError error_code;
};

/**
 * @class BenchmarkTimer
 * @brief High-resolution timer for measuring code execution duration.
 *
 * Provides elapsed time in seconds or milliseconds since construction or last
 * reset.
 */
class BenchmarkTimer {
  std::chrono::high_resolution_clock::time_point start_time_;

public:
  /**
   * @brief Constructs and starts a new timer.
   */
  BenchmarkTimer();

  /**
   * @brief Reset the timer to current time.
   */
  void reset();

  /**
   * @brief Get elapsed time in seconds since last reset or construction.
   * @return Elapsed time in seconds.
   */
  double elapsed_s() const;

  /**
   * @brief Get elapsed time in milliseconds since last reset or construction.
   * @return Elapsed time in milliseconds.
   */
  double elapsed_ms() const;
};

/**
 * @brief Runs a callable multiple times to benchmark its performance.
 *
 * Performs optional warmup iterations, then measures execution time over the
 * specified number of iterations and returns the median duration.
 *
 * @param func_to_run Function or callable that returns true on success.
 * @param iterations Number of measurement iterations (>=1).
 * @param warmup_iterations Number of warmup iterations (>=0).
 * @return BenchmarkResult containing median time and error status.
 */
BenchmarkResult run_benchmark(const std::function<bool()> &func_to_run,
                              int iterations, int warmup_iterations);

} // namespace BenchUtils

#endif // MINIZP_BENCH_UTILS_HPP
