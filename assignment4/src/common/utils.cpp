#include "utils.hpp"
#include <cctype>
#include <climits>
#include <cstring>
#include <iomanip>
#include <stdexcept>

/**
 * @brief Generates a vector of Record objects with specified characteristics.
 *
 * This function creates test data for the sorting algorithm by generating
 * records with keys following different patterns and optional random payload
 * data.
 *
 * @param n The number of records to generate
 * @param payload_size The size of the payload data for each record (in bytes)
 * @param pattern The data pattern to use for key generation:
 *                - RANDOM: Keys are randomly distributed
 *                - SORTED: Keys are in ascending order (0, 1, 2, ...)
 *                - REVERSE_SORTED: Keys are in descending order (n-1, n-2, ...)
 *                - NEARLY_SORTED: Keys are mostly sorted with ~1% disorder
 * @param seed The random seed for reproducible data generation
 *
 * @return std::vector<Record> A vector containing the generated records
 *
 * @note For NEARLY_SORTED pattern, approximately 1% of records will have
 *       their keys swapped with the previous record to introduce disorder.
 * @note If payload_size > 0, each record's payload is filled with random
 *       ASCII characters (0-127).
 */
std::vector<Record> generate_data(size_t n, size_t payload_size,
                                  DataPattern pattern, unsigned seed) {
  std::vector<Record> data;
  data.reserve(n);

  std::mt19937_64 gen(seed);
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
      // Introduce ~1% disorder
      if (dist(gen) % 100 == 0 && i > 0) {
        std::swap(rec.key, data.back().key);
      }
      break;
    }
    if (payload_size > 0) {
      std::uniform_int_distribution<char> char_dist(0, 127);
      for (size_t j = 0; j < payload_size; ++j) {
        rec.payload[j] = char_dist(gen);
      }
    }
    data.push_back(std::move(rec));
  }
  return data;
}

/**
 * @brief Verify array is sorted by key
 */
bool is_sorted(const std::vector<Record> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1].key > data[i].key)
      return false;
  }
  return true;
}

/**
 * @brief Creates a deep copy of a vector of Record objects.
 *
 * This function performs a deep copy of the input vector, creating new Record
 * objects with their own allocated memory for payloads. Each Record in the
 * returned vector will have its own independent copy of the payload data.
 *
 * @param original The source vector of Record objects to be copied
 * @return std::vector<Record> A new vector containing deep copies of all
 * records
 */
std::vector<Record> copy_records(const std::vector<Record> &original) {
  std::vector<Record> copy;
  copy.reserve(original.size());
  for (const auto &rec : original) {
    Record new_rec(rec.payload_size);
    new_rec.key = rec.key;
    if (rec.payload && new_rec.payload && rec.payload_size > 0) {
      std::memcpy(new_rec.payload, rec.payload, rec.payload_size);
    }
    copy.push_back(std::move(new_rec));
  }
  return copy;
}

/**
 * @brief Parse command-line arguments into configuration
 * @return pair<Config, bool> where bool indicates if help was requested
 */
std::pair<Config, bool> parse_args(int argc, char *argv[]) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      return {config, true}; // Help requested
    } else if (arg == "-s" && i + 1 < argc)
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
    } else if (arg == "--csv")
      config.csv_output = true;
    else if (arg == "--csv-file" && i + 1 < argc) {
      config.csv_output = true;
      config.csv_filename = argv[++i];
    }
  }
  return {config, false}; // No help requested
}

/**
 * @brief Format byte count with appropriate units
 */
std::string format_bytes(size_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB"};
  int unit = 0;
  double size = static_cast<double>(bytes);
  while (size >= 1024 && unit < 3) {
    size /= 1024;
    unit++;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
  return oss.str();
}

/**
 * @brief Parse size string with K/M/G suffix
 */
size_t parse_size(const std::string &size_str) {
  if (size_str.empty())
    throw std::invalid_argument("Empty size string");
  std::string str = size_str;
  size_t multiplier = 1;
  char last_char = std::toupper(str.back());
  if (!isdigit(last_char)) {
    str.pop_back();
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
