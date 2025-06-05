#include "utils.hpp"
#include <climits>
#include <cstring>
#include <sstream>
#include <stdexcept>

std::vector<std::unique_ptr<Record>> generate_data(size_t n,
                                                   size_t payload_size,
                                                   DataPattern pattern,
                                                   unsigned seed) {

  std::vector<std::unique_ptr<Record>> data;
  data.reserve(n);

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<unsigned long> dist(0, ULONG_MAX);

  // Generate records
  for (size_t i = 0; i < n; ++i) {
    auto rec = std::make_unique<Record>(payload_size);

    switch (pattern) {
    case DataPattern::RANDOM:
      rec->key = dist(gen);
      break;
    case DataPattern::SORTED:
      rec->key = i;
      break;
    case DataPattern::REVERSE_SORTED:
      rec->key = n - i - 1;
      break;
    case DataPattern::NEARLY_SORTED:
      rec->key = i;
      // Swap ~1% of elements
      if (dist(gen) % 100 == 0 && i > 0) {
        std::swap(rec->key, data[i - 1]->key);
      }
      break;
    }

    // Fill payload with pseudo-random data
    if (payload_size > 0) {
      std::uniform_int_distribution<char> char_dist(0, 127);
      for (size_t j = 0; j < payload_size; ++j) {
        rec->payload[j] = char_dist(gen);
      }
    }

    data.push_back(std::move(rec));
  }

  return data;
}

// Functions for working with Record vectors directly
std::vector<Record> generate_data_vector(size_t n, size_t payload_size,
                                         DataPattern pattern, unsigned seed) {
  std::vector<Record> data;
  data.reserve(n);

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<unsigned long> dist(0, ULONG_MAX);

  // Generate records
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
      // Swap ~1% of elements
      if (dist(gen) % 100 == 0 && i > 0) {
        std::swap(rec.key, data[i - 1].key);
      }
      break;
    }

    // Fill payload with pseudo-random data
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

bool is_sorted(const std::vector<std::unique_ptr<Record>> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1]->key > data[i]->key) {
      return false;
    }
  }
  return true;
}

bool is_sorted_vector(const std::vector<Record> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1].key > data[i].key) {
      return false;
    }
  }
  return true;
}

void print_stats(const std::vector<std::unique_ptr<Record>> &data) {
  if (data.empty())
    return;

  std::cout << "Array size: " << data.size() << " records\n";
  std::cout << "First 5 keys: ";
  for (size_t i = 0; i < std::min(size_t(5), data.size()); ++i) {
    std::cout << data[i]->key << " ";
  }
  std::cout << "\nLast 5 keys: ";
  for (size_t i = std::max(size_t(5), data.size()) - 5; i < data.size(); ++i) {
    std::cout << data[i]->key << " ";
  }
  std::cout << "\nSorted: " << (is_sorted(data) ? "YES" : "NO") << "\n";
}

std::vector<std::unique_ptr<Record>>
copy_records(const std::vector<std::unique_ptr<Record>> &source,
             size_t payload_size) {

  std::vector<std::unique_ptr<Record>> copy;
  copy.reserve(source.size());

  for (const auto &rec : source) {
    auto new_rec = std::make_unique<Record>(payload_size);
    new_rec->key = rec->key;
    if (payload_size > 0 && rec->payload) {
      std::memcpy(new_rec->payload, rec->payload, payload_size);
    }
    copy.push_back(std::move(new_rec));
  }

  return copy;
}

std::vector<Record> copy_records_vector(const std::vector<Record> &original) {
  std::vector<Record> copy;
  copy.reserve(original.size());

  for (const auto &rec : original) {
    // Create a new record with the same data
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

    if (arg == "-s" && i + 1 < argc) {
      config.array_size = std::stoull(argv[++i]);
    } else if (arg == "-r" && i + 1 < argc) {
      config.payload_size = std::stoull(argv[++i]);
    } else if (arg == "-t" && i + 1 < argc) {
      config.num_threads = std::stoull(argv[++i]);
    } else if (arg == "--pattern" && i + 1 < argc) {
      std::string pattern(argv[++i]);
      if (pattern == "random")
        config.pattern = DataPattern::RANDOM;
      else if (pattern == "sorted")
        config.pattern = DataPattern::SORTED;
      else if (pattern == "reverse")
        config.pattern = DataPattern::REVERSE_SORTED;
      else if (pattern == "nearly")
        config.pattern = DataPattern::NEARLY_SORTED;
    } else if (arg == "--no-validate") {
      config.validate = false;
    } else if (arg == "-v" || arg == "--verbose") {
      config.verbose = true;
    }
  }

  return config;
}

std::string format_bytes(size_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB"};
  int unit = 0;
  double size = bytes;

  while (size >= 1024 && unit < 3) {
    size /= 1024;
    unit++;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
  return oss.str();
}
