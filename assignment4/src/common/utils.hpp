#ifndef UTILS_HPP
#define UTILS_HPP

#include "record.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

enum class DataPattern { RANDOM, SORTED, REVERSE_SORTED, NEARLY_SORTED };

/**
 * @brief Generate array of records with specified pattern
 */
std::vector<std::unique_ptr<Record>>
generate_data(size_t n, size_t payload_size,
              DataPattern pattern = DataPattern::RANDOM,
              unsigned seed = std::random_device{}());

/**
 * @brief Verify array is sorted correctly
 */
bool is_sorted(const std::vector<std::unique_ptr<Record>> &data);

/**
 * @brief Print statistics about the dataset
 */
void print_stats(const std::vector<std::unique_ptr<Record>> &data);

/**
 * @brief Deep copy of record array (for testing)
 */
std::vector<std::unique_ptr<Record>>
copy_records(const std::vector<std::unique_ptr<Record>> &source,
             size_t payload_size);

/**
 * @brief Parse command line arguments
 */
struct Config {
  size_t array_size = 1000000; // -s
  size_t payload_size = 8;     // -r
  size_t num_threads = 4;      // -t
  DataPattern pattern = DataPattern::RANDOM;
  bool validate = true;
  bool verbose = false;
};

Config parse_args(int argc, char *argv[]);

/**
 * @brief Format bytes for human reading
 */
std::string format_bytes(size_t bytes);

/**
 * @brief Generate array of records as direct vectors (not unique_ptr)
 */
std::vector<Record>
generate_data_vector(size_t n, size_t payload_size,
                     DataPattern pattern = DataPattern::RANDOM,
                     unsigned seed = std::random_device{}());

/**
 * @brief Verify array is sorted correctly (direct vector version)
 */
bool is_sorted_vector(const std::vector<Record> &data);

/**
 * @brief Deep copy of record array (direct vector version)
 */
std::vector<Record> copy_records_vector(const std::vector<Record> &original);

#endif // UTILS_HPP
