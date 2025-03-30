#ifndef TESTING_H
#define TESTING_H

#include "common_types.h"
#include <string>
#include <vector>

/**
 * @brief Runs the comprehensive correctness test suite.
 * Compares parallel implementations against sequential implementation
 * for various test cases including edge cases.
 *
 * @return true if all tests pass, false otherwise.
 */
bool run_correctness_suite();

/**
 * @brief Runs performance tests comparing static schedulers.
 * Results are saved to static_comparison.csv.
 *
 * @param thread_counts Thread counts to test
 * @param chunk_sizes Chunk sizes to test
 * @param samples Number of samples per configuration
 * @param iterations_per_sample Iterations per sample
 * @param workload Workload ranges
 * @return true if tests run successfully
 */
bool run_static_performance_comparison(const std::vector<int> &thread_counts,
                                       const std::vector<ull> &chunk_sizes,
                                       int samples, int iterations_per_sample,
                                       const std::vector<Range> &workload);

/**
 * @brief Runs performance tests comparing all scheduler implementations.
 * Uses sequential implementation as baseline for speedup.
 * Results are saved to all_schedulers.csv.
 *
 * @param thread_counts Thread counts to test
 * @param chunk_sizes Chunk sizes to test
 * @param samples Number of samples per configuration
 * @param iterations_per_sample Iterations per sample
 * @param workload Workload ranges
 * @return true if tests run successfully
 */
bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload);

/**
 * @brief Runs additional performance tests with varying workloads.
 * Results are saved to workload_scaling.csv.
 *
 * @param thread_counts Thread counts to test
 * @param workloads Different workloads to test
 * @param samples Number of samples per configuration
 * @param iterations_per_sample Iterations per sample
 * @return true if tests run successfully
 */
bool run_workload_scaling_tests(
    const std::vector<int> &thread_counts,
    const std::vector<std::vector<Range>> &workloads, int samples,
    int iterations_per_sample);

#endif // TESTING_H
