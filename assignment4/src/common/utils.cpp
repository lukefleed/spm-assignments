// FILE: src/common/utils.cpp

#include "utils.hpp"
#include <cctype>
#include <climits>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread> // Required for hardware_concurrency

// Implementations for non-namespaced functions remain the same
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

bool is_sorted(const std::vector<Record> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1].key > data[i].key)
      return false;
  }
  return true;
}

void print_stats(const std::vector<Record> &data) {
  if (data.empty()) {
    std::cout << "Empty dataset\n";
    return;
  }
  std::cout << "Array size: " << data.size() << " records\n";
  std::cout << "Payload size: " << data[0].payload_size << " bytes\n";
  std::cout << "Total memory: "
            << format_bytes(data.size() *
                            (sizeof(unsigned long) + data[0].payload_size))
            << "\n";
  std::cout << "First 5 keys: ";
  for (size_t i = 0; i < std::min(size_t(5), data.size()); ++i)
    std::cout << data[i].key << " ";
  std::cout << "\nLast 5 keys: ";
  size_t start = data.size() >= 5 ? data.size() - 5 : 0;
  for (size_t i = start; i < data.size(); ++i)
    std::cout << data[i].key << " ";
  std::cout << "\nSorted: " << (is_sorted(data) ? "YES" : "NO") << "\n";
}

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

std::vector<Record> generate_records(size_t n, size_t payload_size,
                                     DataPattern pattern, unsigned seed) {
  return generate_data(n, payload_size, pattern, seed);
}

// Implementation for the new namespaced function
namespace utils {

size_t get_optimal_parallel_threads() {
  // Get hardware concurrency, with a fallback.
  size_t hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) {
    hw_threads = 4; // A reasonable fallback for virtualized environments.
  }
  // Use 75% of available cores to leave resources for MPI and OS tasks.
  // Ensure at least 1 thread is returned.
  return std::max(size_t(1), static_cast<size_t>(hw_threads * 0.75));
}

} // namespace utils
