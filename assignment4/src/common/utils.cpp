/**
 * @file utils.cpp
 * @brief Utility functions for parallel sorting benchmarks and data generation.
 */

#include "utils.hpp"
#include <cctype>
#include <climits>
#include <cstring>
#include <sstream>
#include <stdexcept>

/**
 * @brief Generates test dataset with specified pattern and payload.
 * @details Optimized for cache efficiency using reserve() and move semantics.
 *          NEARLY_SORTED introduces 1% disorder for realistic scenarios.
 * @param n Number of records to generate
 * @param payload_size Size of each record's payload in bytes
 * @param pattern Data distribution pattern for benchmarking
 * @param seed RNG seed for reproducible results
 * @return Vector of generated records
 */
std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern, unsigned seed) {
  std::vector<Record> data;
  data.reserve(n); // Pre-allocate to avoid reallocations during generation

  std::mt19937_64 gen(seed); // 64-bit Mersenne Twister for quality randomness
  std::uniform_int_distribution<unsigned long> dist(0, ULONG_MAX);

  for (size_t i = 0; i < n; ++i) {
    Record rec(payload_size);
    switch (pattern) {
    case DataPattern::RANDOM:
      rec.key = dist(gen);
      break;
    case DataPattern::SORTED:
      rec.key = i;
      break;
    case DataPattern::REVERSE_SORTED:
      rec.key = n - i - 1;
      break;
    case DataPattern::NEARLY_SORTED:
      rec.key = i;
      // Introduce ~1% disorder by swapping adjacent elements
      if (dist(gen) % 100 == 0 && i > 0) {
        std::swap(rec.key, data.back().key);
      }
      break;
    }
    if (payload_size > 0) {
      // ASCII characters only for portable payload generation
      std::uniform_int_distribution<char> char_dist(0, 127);
      for (size_t j = 0; j < payload_size; ++j) {
        rec.payload[j] = char_dist(gen);
      }
    }
    data.push_back(std::move(rec)); // Move to avoid unnecessary copy
  }
  return data;
}

/**
 * @brief Verifies array is sorted in ascending order by key.
 * @details Single-pass O(n) verification with early termination.
 * @param data Vector of records to check
 * @return true if sorted, false otherwise
 */

bool is_sorted(const std::vector<Record> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1].key > data[i].key)
      return false;
  }
  return true;
}

/**
 * @brief Creates deep copy of record vector with payload duplication.
 * @details Memory-safe copying using memcpy for payload data.
 * @param original Source vector to copy
 * @return Independent copy of the original vector
 */

std::vector<Record> copy_records(const std::vector<Record> &original) {
  std::vector<Record> copy;
  copy.reserve(original.size()); // Pre-allocate for efficiency
  for (const auto &rec : original) {
    Record new_rec(rec.payload_size);
    new_rec.key = rec.key;
    // Safe payload copying with null checks
    if (rec.payload && new_rec.payload && rec.payload_size > 0) {
      std::memcpy(new_rec.payload, rec.payload, rec.payload_size);
    }
    copy.push_back(std::move(new_rec));
  }
  return copy;
}

/**
 * @brief Parses command-line arguments into configuration structure.
 * @details Basic argument parser without error handling for missing values.
 * @param argc Argument count
 * @param argv Argument vector
 * @return Populated configuration structure
 */

Config parse_args(int argc, char *argv[]) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-s" && i + 1 < argc)
      config.array_size = parse_size(argv[++i]);
    else if (arg == "-r" && i + 1 < argc)
      config.payload_size = std::stoul(argv[++i]);
    else if (arg == "-t" && i + 1 < argc)
      config.num_threads = std::stoul(argv[++i]);
    else if (arg == "--pattern" && i + 1 < argc) {
      std::string p(argv[++i]);
      if (p == "random")
        config.pattern = DataPattern::RANDOM;
      else if (p == "sorted")
        config.pattern = DataPattern::SORTED;
      else if (p == "reverse")
        config.pattern = DataPattern::REVERSE_SORTED;
      else if (p == "nearly")
        config.pattern = DataPattern::NEARLY_SORTED;
    } else if (arg == "--no-validate")
      config.validate = false;
    else if (arg == "-v" || arg == "--verbose")
      config.verbose = true;
  }
  return config;
}

/**
 * @brief Formats byte count into human-readable string with appropriate units.
 * @details Uses binary units (1024-based) with 2 decimal precision.
 * @param bytes Raw byte count to format
 * @return Formatted string with units (B, KB, MB, GB)
 */

std::string format_bytes(size_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB"};
  int unit = 0;
  double size = static_cast<double>(bytes);
  // Convert to larger units while maintaining precision
  while (size >= 1024 && unit < 3) {
    size /= 1024;
    unit++;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
  return oss.str();
}

/**
 * @brief Parses size string with optional K/M/G suffix into byte count.
 * @details Supports binary multipliers (1024-based) with input validation.
 * @param size_str Input string (e.g., "100M", "1G", "512")
 * @return Size in bytes
 * @throws std::invalid_argument for malformed input
 */

size_t parse_size(const std::string &size_str) {
  if (size_str.empty())
    throw std::invalid_argument("Empty size string");
  std::string str = size_str;
  size_t multiplier = 1;
  char last_char = std::toupper(str.back());
  if (!isdigit(last_char)) {
    str.pop_back();
    // Binary multipliers for memory-related calculations
    if (last_char == 'K')
      multiplier = 1024;
    else if (last_char == 'M')
      multiplier = 1024 * 1024;
    else if (last_char == 'G')
      multiplier = 1024 * 1024 * 1024;
    else
      throw std::invalid_argument("Invalid size suffix");
  }
  try {
    return std::stoull(str) * multiplier;
  } catch (const std::exception &) {
    throw std::invalid_argument("Invalid size format");
  }
}

namespace utils {} // namespace utils
