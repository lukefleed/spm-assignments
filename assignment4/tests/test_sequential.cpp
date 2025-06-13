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
  Config config = parse_args(argc, argv);

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
    if (config.validate && !is_sorted(data_copy)) {
      std::cerr << "ERROR: MergeSort failed validation!\n";
      return 1;
    }
  }

  // Test std::sort baseline
  {
    auto data_copy = copy_records(data);
    Timer t("std::sort");
    stl_sort(data_copy);
    double ms = t.elapsed_ms();

    std::cout << "std::sort: " << ms << " ms\n";
    if (config.validate && !is_sorted(data_copy)) {
      std::cerr << "ERROR: std::sort failed validation!\n";
      return 1;
    }
  }

  return 0;
}
