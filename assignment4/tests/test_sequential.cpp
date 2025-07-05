#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <iostream>

/**
 * @brief std::sort adapter for consistent interface
 */
void stl_sort(std::vector<Record> &data) {
  std::sort(data.begin(), data.end());
}

/**
 * @brief Sequential sorting benchmark and validation
 */
int main(int argc, char *argv[]) {
  auto [config, help_requested] = parse_args(argc, argv);
  if (help_requested) {
    std::cout << "Usage: test_sequential [OPTIONS]\n\n";
    std::cout << "Sequential sorting benchmark and validation.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -s SIZE                 Array size (supports K/M/G suffixes, default: 1000000)\n";
    std::cout << "  -r SIZE                 Record payload size in bytes (default: 8)\n";
    std::cout << "  --pattern PATTERN       Data pattern: random, sorted, reverse, nearly (default: random)\n\n";
    return 0;
  }

  std::cout << "Testing sequential sort implementations\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n\n";

  // Generate test data
  auto data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  // Test custom sequential mergesort
  {
    auto data_copy = copy_records(data);
    Timer t("Sequential MergeSort");
    sequential_mergesort(data_copy);
    double ms = t.elapsed_ms();

    std::cout << "Sequential MergeSort: " << ms << " ms\n";
  }

  // Test std::sort baseline
  {
    auto data_copy = copy_records(data);
    Timer t("std::sort");
    stl_sort(data_copy);
    double ms = t.elapsed_ms();

    std::cout << "std::sort: " << ms << " ms\n";
  }

  return 0;
}
