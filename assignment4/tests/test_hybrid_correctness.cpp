/**
 * @file test_hybrid_correctness.cpp
 * @brief Correctness testing for hybrid MPI+parallel mergesort
 */

#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/hybrid/mpi_ff_mergesort.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>

/**
 * @brief Test case structure
 */
struct TestCase {
  size_t data_size;
  size_t payload_size;
  DataPattern pattern;
  size_t parallel_threads;
  std::string description;
};

/**
 * @brief Verify that data is correctly sorted
 */
bool verify_sorted(const std::vector<Record> &data) {
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1].key > data[i].key) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Main test function
 */
int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "MPI implementation does not support required threading level"
              << std::endl;
    MPI_Finalize();
    return 1;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\nHybrid MPI+Parallel Mergesort Correctness Tests\n";
    std::cout << "===============================================\n";
    std::cout << "Running with " << size << " MPI processes\n\n";
  }

  // Define test cases using the same pattern as multi_node_main
  std::vector<TestCase> test_cases = {
      {1000, 0, DataPattern::RANDOM, 2, "Small random data"},
      {1000, 8, DataPattern::RANDOM, 2, "Small data with payload"},
      {1000, 0, DataPattern::SORTED, 2, "Already sorted data"},
      {1000, 0, DataPattern::REVERSE_SORTED, 2, "Reverse sorted data"},
      {10000, 0, DataPattern::RANDOM, 4, "Medium random data"},
      {50000, 0, DataPattern::RANDOM, 8, "Large random data"},
      {100, 128, DataPattern::RANDOM, 2, "Large payload test"}};

  int passed = 0;
  int failed = 0;

  for (const auto &test : test_cases) {
    if (rank == 0) {
      std::cout << "Testing " << test.description << " (size=" << test.data_size
                << ", payload=" << test.payload_size
                << ", threads=" << test.parallel_threads << ")... ";
      std::cout.flush();
    }

    Timer timer;
    bool success = true;

    try {
      // Generate test data using the same method as multi_node_main
      auto data =
          generate_data(test.data_size, test.payload_size, test.pattern);

      // Create copy for validation
      std::vector<uint64_t> original_keys;
      for (const auto &record : data) {
        original_keys.push_back(record.key);
      }
      std::sort(original_keys.begin(), original_keys.end());

      // Run hybrid sort using the same class as multi_node_main
      hybrid::HybridConfig config;
      config.parallel_threads = test.parallel_threads;
      hybrid::HybridMergeSort sorter(config);
      auto result = sorter.sort(data, test.payload_size);

      // Verify on rank 0
      if (rank == 0) {
        // Check if sorted
        if (!verify_sorted(result)) {
          success = false;
        } else {
          // Check if same elements
          std::vector<uint64_t> result_keys;
          for (const auto &record : result) {
            result_keys.push_back(record.key);
          }
          std::sort(result_keys.begin(), result_keys.end());

          if (original_keys != result_keys) {
            success = false;
          }
        }
      }

    } catch (const std::exception &e) {
      success = false;
      if (rank == 0) {
        std::cout << "Exception: " << e.what() << "\n";
      }
    }

    // Broadcast result to all processes
    int success_int = success ? 1 : 0;
    MPI_Bcast(&success_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    success = (success_int == 1);

    double elapsed = timer.elapsed_ms();

    if (rank == 0) {
      if (success) {
        std::cout << "PASSED";
        passed++;
      } else {
        std::cout << "FAILED";
        failed++;
      }
      std::cout << " (" << std::fixed << std::setprecision(2) << elapsed
                << " ms)\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::cout << "\n===============================================\n";
    std::cout << "Test Summary:\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Success Rate: " << std::fixed << std::setprecision(1)
              << (100.0 * passed / (passed + failed)) << "%\n";

    if (failed == 0) {
      std::cout << "\nAll tests passed! Hybrid implementation is correct.\n";
    } else {
      std::cout << "\nSome tests failed. Please review implementation.\n";
    }
  }

  MPI_Finalize();
  return failed == 0 ? 0 : 1;
}
