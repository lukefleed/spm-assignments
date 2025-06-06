#ifndef UTILS_HPP
#define UTILS_HPP

#include "record.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

enum class DataPattern { RANDOM, SORTED, REVERSE_SORTED, NEARLY_SORTED };

/**
 * @brief Generate array of records with specified pattern
 * @param n Number of records to generate
 * @param payload_size Size of payload in bytes
 * @param pattern Data distribution pattern
 * @param seed Random seed for reproducibility
 */
std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern = DataPattern::RANDOM,
                                  unsigned seed = std::random_device{}());

/**
 * @brief Verify array is sorted correctly by key
 * @param data Vector of records to check
 * @return true if sorted in ascending order by key
 */
bool is_sorted(const std::vector<Record> &data);

/**
 * @brief Print dataset statistics and verification results
 * @param data Vector of records to analyze
 */
void print_stats(const std::vector<Record> &data);

/**
 * @brief Create deep copy of record vector for testing
 * @param original Source vector to copy
 * @return Independent copy with same data
 */
std::vector<Record> copy_records(const std::vector<Record> &original);

/**
 * @brief Command line configuration structure
 */
struct Config {
  size_t array_size = 1000000; // -s
  size_t payload_size = 8;     // -r
  size_t num_threads = 4;      // -t
  DataPattern pattern = DataPattern::RANDOM;
  bool validate = true;
  bool verbose = false;
};

/**
 * @brief Parse command line arguments into configuration
 * @param argc Argument count
 * @param argv Argument values
 * @return Parsed configuration structure
 */
Config parse_args(int argc, char *argv[]);

/**
 * @brief Format byte count for human-readable display
 * @param bytes Number of bytes
 * @return Formatted string with appropriate units
 */
std::string format_bytes(size_t bytes);

#endif // UTILS_HPP
