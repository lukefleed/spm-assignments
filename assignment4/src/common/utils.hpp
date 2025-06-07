// FILE: src/common/utils.hpp

#ifndef UTILS_HPP
#define UTILS_HPP

#include "record.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

enum class DataPattern { RANDOM, SORTED, REVERSE_SORTED, NEARLY_SORTED };

std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern = DataPattern::RANDOM,
                                  unsigned seed = std::random_device{}());

bool is_sorted(const std::vector<Record> &data);
void print_stats(const std::vector<Record> &data);
std::vector<Record> copy_records(const std::vector<Record> &original);

struct Config {
  size_t array_size = 1000000;
  size_t payload_size = 8;
  size_t num_threads = 4;
  DataPattern pattern = DataPattern::RANDOM;
  bool validate = true;
  bool verbose = false;
};

Config parse_args(int argc, char *argv[]);
std::string format_bytes(size_t bytes);
size_t parse_size(const std::string &size_str);
std::vector<Record> generate_records(size_t n, size_t payload_size,
                                     DataPattern pattern = DataPattern::RANDOM,
                                     unsigned seed = std::random_device{}());

/**
 * @brief Namespace for utility functions related to parallel execution.
 */
namespace utils {

/**
 * @brief Determines optimal parallel worker thread count based on system
 * resources.
 * @details Calculates 75% of available CPU cores to reserve resources for
 *          MPI communication and system processes, preventing contention.
 * @return Recommended thread count for parallel execution framework.
 */
size_t get_optimal_parallel_threads();

} // namespace utils

#endif // UTILS_HPP
