#ifndef MINIZP_BENCH_UTILS_HPP
#define MINIZP_BENCH_UTILS_HPP

#include <chrono> // For high_resolution_clock (alternative to omp_get_wtime)
#include <functional> // For std::function
#include <omp.h>      // For omp_get_wtime
#include <vector>

namespace BenchUtils {

/**
 * @brief Simple timer using omp_get_wtime.
 */
class BenchmarkTimer {
  double start_time_;

public:
  BenchmarkTimer() : start_time_(omp_get_wtime()) {}

  /** @brief Resets the timer start time. */
  void reset() { start_time_ = omp_get_wtime(); }

  /** @brief Returns elapsed time in seconds since construction or last reset.
   */
  double elapsed_s() const { return omp_get_wtime() - start_time_; }

  /** @brief Returns elapsed time in milliseconds since construction or last
   * reset. */
  double elapsed_ms() const { return elapsed_s() * 1000.0; }
};

/**
 * @brief Runs a function multiple times and returns the median execution time.
 *
 * @param func_to_run The function to benchmark. It should return true on
 * success, false on error.
 * @param iterations The number of measurement iterations. Must be >= 1.
 * @param warmup_iterations The number of warmup iterations (not measured). Must
 * be >= 0.
 * @return double The median execution time in seconds. Returns a negative value
 * on error or if func_to_run fails.
 */
double run_benchmark(const std::function<bool()> &func_to_run, int iterations,
                     int warmup_iterations);

} // namespace BenchUtils

#endif // MINIZP_BENCH_UTILS_HPP
