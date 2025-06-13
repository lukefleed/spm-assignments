#ifndef UTILS_HPP
#define UTILS_HPP

#include "record.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief Data distribution patterns for algorithm stress testing
 */
enum class DataPattern {
  RANDOM,         ///< Uniformly distributed random keys
  SORTED,         ///< Ascending order
  REVERSE_SORTED, ///< Descending order
  NEARLY_SORTED   ///< ~99% sorted with random disorder
};

/**
 * @brief Generate test dataset with specified pattern
 * @param n Number of records
 * @param payload_size Payload size in bytes
 * @param pattern Data distribution pattern
 * @param seed RNG seed for reproducible results
 * @return Vector of generated records
 */
std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern = DataPattern::RANDOM,
                                  unsigned seed = std::random_device{}());

/**
 * @brief Verify array is sorted by key
 */
bool is_sorted(const std::vector<Record> &data);

/**
 * @brief Deep copy record vector with payload duplication
 */
std::vector<Record> copy_records(const std::vector<Record> &original);

/**
 * @brief Benchmark configuration parameters
 */
struct Config {
  size_t array_size = 1000000;
  size_t payload_size = 8;
  size_t num_threads = 4;
  DataPattern pattern = DataPattern::RANDOM;
  bool validate = true;
  bool verbose = false;
  bool csv_output = false;
  std::string csv_filename = "";
};

/**
 * @brief Parse command-line arguments into configuration
 */
Config parse_args(int argc, char *argv[]);

/**
 * @brief Format byte count with appropriate units (B, KB, MB, GB)
 */
std::string format_bytes(size_t bytes);

/**
 * @brief Parse size string with K/M/G suffix
 * @param size_str Input string (e.g., "100M", "1G", "512")
 * @return Size in bytes
 * @throws std::invalid_argument for invalid input
 */
size_t parse_size(const std::string &size_str);

#endif // UTILS_HPP
