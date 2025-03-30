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
 * @brief Runs the main performance benchmark suite.
 * Tests various schedulers (sequential, static types, dynamic) across
 * different workloads, thread counts, and chunk sizes.
 * Calculates speedup relative to the sequential execution of each specific
 * workload. Saves detailed results to a CSV file.
 *
 * @param thread_counts Thread counts to test for parallel schedulers.
 * @param chunk_sizes Chunk sizes to test for relevant schedulers.
 * @param workloads A list of different workloads (each a vector of Ranges) to
 * test.
 * @param workload_descriptions Corresponding descriptions for each workload.
 * @param samples Number of samples for time measurement per configuration.
 * @param iterations_per_sample Number of iterations within each sample.
 * @return true if the benchmark suite runs successfully, false otherwise.
 */
bool run_benchmark_suite(const std::vector<int> &thread_counts,
                         const std::vector<ull> &chunk_sizes,
                         const std::vector<std::vector<Range>> &workloads,
                         const std::vector<std::string> &workload_descriptions,
                         int samples, int iterations_per_sample);

#endif // TESTING_H
