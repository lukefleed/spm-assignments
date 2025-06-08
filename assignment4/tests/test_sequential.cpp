/**
 * @file test_sequential.cpp
 * @brief Performance benchmarking and validation for sequential mergesort
 * implementation
 *
 * Comparative testing framework evaluating custom sequential mergesort against
 * std::sort baseline. Provides isolated performance measurements and optional
 * correctness validation across configurable data patterns and sizes.
 */

#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <iostream>

/**
 * @brief Adapter for std::sort maintaining consistent interface with custom
 * implementation
 * @param data Record vector to sort in-place
 *
 * Provides interface consistency for comparative benchmarking. std::sort serves
 * as performance baseline due to its highly optimized introsort implementation
 * (quicksort + heapsort + insertion sort hybrid).
 */
void stl_sort(std::vector<Record> &data) {
  std::sort(data.begin(), data.end());
}

/**
 * @brief Sequential sorting benchmark and validation entry point
 * @param argc Command line argument count
 * @param argv Command line arguments for test configuration
 * @return 0 on success, 1 on validation failure
 *
 * Executes isolated performance comparison between custom sequential mergesort
 * and std::sort baseline. Independent data copies prevent cross-contamination
 * between test runs. Timer overhead is minimized through RAII scoping.
 *
 * Validation is conditionally performed based on configuration to enable
 * pure performance benchmarking when correctness is already established.
 */
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "Testing sequential sort implementations\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n\n";

  // Generate test data
  auto data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  // Test custom mergesort
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

  // Test std::sort
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
