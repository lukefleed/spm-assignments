/**
 * @file utils.hpp
 * @brief Utility functions and types for parallel sorting benchmark framework.
 * @details Provides data generation, configuration parsing, and system resource
 *          management for high-performance parallel sorting algorithms.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include "record.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief Data distribution patterns for benchmark testing.
 * @details Different patterns stress-test sorting algorithms under varying
 *          input conditions to evaluate performance characteristics.
 */
enum class DataPattern {
  RANDOM,         ///< Uniformly distributed random keys
  SORTED,         ///< Already sorted in ascending order (best case)
  REVERSE_SORTED, ///< Sorted in descending order (worst case)
  NEARLY_SORTED   ///< ~99% sorted with 1% disorder (realistic scenario)
};

/**
 * @brief Generates test dataset with specified pattern and payload.
 * @param n Number of records to generate
 * @param payload_size Size of each record's payload in bytes
 * @param pattern Data distribution pattern for benchmarking
 * @param seed RNG seed for reproducible results
 * @return Vector of generated records with specified characteristics
 */
std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern = DataPattern::RANDOM,
                                  unsigned seed = std::random_device{}());

/**
 * @brief Verifies array is sorted in ascending order by key.
 * @param data Vector of records to check
 * @return true if sorted, false otherwise
 */
bool is_sorted(const std::vector<Record> &data);

/**
 * @brief Outputs comprehensive dataset statistics for verification.
 * @param data Vector of records to analyze
 */
void print_stats(const std::vector<Record> &data);

/**
 * @brief Creates deep copy of record vector with payload duplication.
 * @param original Source vector to copy
 * @return Independent copy of the original vector
 */
std::vector<Record> copy_records(const std::vector<Record> &original);

/**
 * @brief Configuration parameters for sorting benchmarks.
 * @details Encapsulates all tunable parameters with performance-optimized
 *          defaults suitable for typical multi-core systems.
 */
struct Config {
  size_t array_size = 1000000; ///< Records count (1M default for L3 cache fit)
  size_t payload_size = 8;     ///< Payload bytes per record (cache-friendly)
  size_t num_threads = 4;      ///< Worker thread count (conservative default)
  DataPattern pattern =
      DataPattern::RANDOM; ///< Input data distribution pattern
  bool validate = true;    ///< Enable correctness verification
  bool verbose = false;    ///< Enable detailed output logging
};

/**
 * @brief Parses command-line arguments into configuration structure.
 * @param argc Argument count
 * @param argv Argument vector
 * @return Populated configuration structure
 */
Config parse_args(int argc, char *argv[]);

/**
 * @brief Formats byte count into human-readable string with appropriate units.
 * @param bytes Raw byte count to format
 * @return Formatted string with binary units (B, KB, MB, GB)
 */
std::string format_bytes(size_t bytes);

/**
 * @brief Parses size string with optional K/M/G suffix into byte count.
 * @param size_str Input string (e.g., "100M", "1G", "512")
 * @return Size in bytes
 * @throws std::invalid_argument for malformed input
 */
size_t parse_size(const std::string &size_str);

/**
 * @brief System resource management utilities for parallel execution.
 * @details Provides optimal thread count calculation for MPI+threading
 *          hybrid parallelization strategies.
 */
namespace utils {

/**
 * @brief Determines optimal parallel worker thread count based on system
 * resources.
 * @details Calculates 75% of available CPU cores to reserve resources for
 *          MPI communication and system processes, preventing contention.
 *          Minimum return value is 1, with fallback for virtualized
 * environments.
 * @return Recommended thread count for parallel execution framework
 */
size_t get_optimal_parallel_threads();

} // namespace utils

#endif // UTILS_HPP
