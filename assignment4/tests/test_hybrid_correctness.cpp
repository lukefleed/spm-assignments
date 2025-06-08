/**
 * @file test_hybrid_correctness.cpp
 * @brief Comprehensive correctness validation for hybrid MPI+FastFlow mergesort
 * implementation
 *
 * Tests distributed sorting across multiple data patterns and configurations.
 * Requires MPI_THREAD_FUNNELED threading level for hybrid parallel execution.
 *
 * @note All processes participate in sorting; only rank 0 performs validation
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
 * @brief Test configuration for hybrid sorting validation
 */
struct TestCase {
  size_t data_size;        ///< Number of records to sort
  size_t payload_size;     ///< Additional payload bytes per record
  DataPattern pattern;     ///< Initial data distribution pattern
  size_t parallel_threads; ///< FastFlow parallelism within each MPI process
  std::string description; ///< Human-readable test description
};

/**
 * @brief Validates sorting correctness using linear scan
 * @param data Sorted record vector to validate
 * @return true if data is properly sorted by key in ascending order
 *
 * O(n) verification chosen over binary search approach for simplicity
 * and predictable cache behavior in test environment.
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
 * @brief Executes comprehensive correctness tests for hybrid MPI+FastFlow
 * mergesort
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return 0 if all tests pass, 1 if any test fails
 *
 * Coordinates distributed testing across MPI processes with rank 0 performing
 * result validation. Each test verifies both ordering correctness and element
 * preservation across the distributed sort operation.
 */
int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  // Validate MPI threading support for hybrid execution
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

  // Test suite covers edge cases: small/large data, various patterns, payload
  // sizes
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
      // Generate test data
      auto data =
          generate_data(test.data_size, test.payload_size, test.pattern);

      // Store original key distribution for element preservation verification
      std::vector<uint64_t> original_keys;
      for (const auto &record : data) {
        original_keys.push_back(record.key);
      }
      std::sort(original_keys.begin(), original_keys.end());

      // Execute distributed hybrid sort with specified thread configuration
      hybrid::HybridConfig config;
      config.parallel_threads = test.parallel_threads;
      hybrid::HybridMergeSort sorter(config);
      auto result = sorter.sort(data, test.payload_size);

      // Centralized validation on rank 0 to avoid coordination overhead
      if (rank == 0) {
        // Primary correctness check: verify ascending order
        if (!verify_sorted(result)) {
          success = false;
        } else {
          // Secondary check: ensure no elements lost or duplicated during
          // distribution
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

    // Synchronize test result across all processes for consistent state
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

    // Ensure all processes complete test before proceeding to next
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
