/**
 * @file test_correctness.cpp
 * @brief Comprehensive correctness validation for mergesort implementations
 *
 * Validates both sequential and parallel mergesort algorithms through
 * extensive test cases covering edge conditions, data patterns, and
 * cross-implementation consistency verification.
 */

#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

/**
 * @brief Validates sorted array integrity and ordering correctness
 *
 * Performs dual validation: ordering verification (O(n)) and key preservation
 * check using multiset comparison (O(n log n)). Multiset approach chosen over
 * sorting original data to avoid modifying input and enable concurrent testing.
 *
 * @param sorted Array to validate for correct ordering
 * @param original Reference array for key preservation verification
 * @return true if array is properly sorted and contains identical key set
 */
bool validate_sort_correctness(const std::vector<Record> &sorted,
                               const std::vector<Record> &original) {
  if (sorted.size() != original.size()) {
    std::cerr << "Size mismatch: expected " << original.size() << ", got "
              << sorted.size() << std::endl;
    return false;
  }

  // Verify ascending order invariant
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i - 1].key > sorted[i].key) {
      std::cerr << "Not sorted at position " << i << ": " << sorted[i - 1].key
                << " > " << sorted[i].key << std::endl;
      return false;
    }
  }

  // Verify key preservation via multiset equality check
  std::multiset<unsigned long> original_keys, sorted_keys;
  for (const auto &record : original) {
    original_keys.insert(record.key);
  }
  for (const auto &record : sorted) {
    sorted_keys.insert(record.key);
  }

  if (original_keys != sorted_keys) {
    std::cerr << "Key content mismatch after sorting" << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief Test case configuration encapsulating all test parameters
 */
struct TestCase {
  std::string name;    ///< Human-readable test identifier
  size_t size;         ///< Number of records to sort
  size_t payload_size; ///< Record payload size in bytes
  DataPattern pattern; ///< Initial data ordering pattern
  size_t threads;      ///< Thread count for parallel implementation
};

/**
 * @brief Executes comprehensive test validation with cross-implementation
 * verification
 *
 * Performs isolated testing of both sequential and parallel implementations
 * using independent data copies to prevent interference. Validates each
 * implementation against original data, then cross-validates for result
 * equivalence to detect implementation-specific issues or non-deterministic
 * behavior.
 *
 * @param test Test configuration specifying size, pattern, and threading
 * parameters
 * @return true if both implementations pass validation and produce equivalent
 * results
 */
bool run_test_case(const TestCase &test) {
  std::cout << "Running test: " << test.name << " (size=" << test.size
            << ", threads=" << test.threads << ")..." << std::flush;

  // Generate test data with specified characteristics
  auto original_data =
      generate_data(test.size, test.payload_size, test.pattern);

  // Test sequential implementation with independent data copy
  auto seq_data = copy_records(original_data);
  sequential_mergesort(seq_data);

  if (!validate_sort_correctness(seq_data, original_data)) {
    std::cout << " FAILED (Sequential)" << std::endl;
    return false;
  }

  // Test parallel implementation with independent data copy
  auto ff_data = copy_records(original_data);
  parallel_mergesort(ff_data, test.threads);

  if (!validate_sort_correctness(ff_data, original_data)) {
    std::cout << " FAILED (Parallel)" << std::endl;
    return false;
  }

  // Cross-validate implementation consistency
  if (seq_data.size() != ff_data.size()) {
    std::cout << " FAILED (Size mismatch between implementations)" << std::endl;
    return false;
  }

  for (size_t i = 0; i < seq_data.size(); ++i) {
    if (seq_data[i].key != ff_data[i].key) {
      std::cout << " FAILED (Result mismatch at position " << i << ")"
                << std::endl;
      return false;
    }
  }

  std::cout << " PASSED" << std::endl;
  return true;
}

int main() {
  std::cout << "=== Single Node MergeSort Correctness Tests ===" << std::endl;

  std::vector<TestCase> test_cases = {
      // Boundary conditions and degenerate cases
      {"Empty array", 0, 8, DataPattern::RANDOM, 1},
      {"Single element", 1, 8, DataPattern::RANDOM, 1},
      {"Two elements (random)", 2, 8, DataPattern::RANDOM, 1},
      {"Two elements (sorted)", 2, 8, DataPattern::SORTED, 1},
      {"Two elements (reverse)", 2, 8, DataPattern::REVERSE_SORTED, 1},

      // Small dataset validation across thread configurations
      {"Small random (thread=1)", 10, 8, DataPattern::RANDOM, 1},
      {"Small random (thread=2)", 10, 8, DataPattern::RANDOM, 2},
      {"Small sorted", 10, 8, DataPattern::SORTED, 2},
      {"Small reverse", 10, 8, DataPattern::REVERSE_SORTED, 2},
      {"Small nearly sorted", 10, 8, DataPattern::NEARLY_SORTED, 2},

      // Medium-scale pattern-specific validation
      {"Medium random", 1000, 8, DataPattern::RANDOM, 4},
      {"Medium sorted", 1000, 8, DataPattern::SORTED, 4},
      {"Medium reverse", 1000, 8, DataPattern::REVERSE_SORTED, 4},
      {"Medium nearly sorted", 1000, 8, DataPattern::NEARLY_SORTED, 4},

      // Large-scale stress testing
      {"Large random", 100000, 8, DataPattern::RANDOM, 8},
      {"Large sorted", 100000, 8, DataPattern::SORTED, 8},
      {"Large reverse", 100000, 8, DataPattern::REVERSE_SORTED, 8},

      // Variable payload size impact assessment
      {"Small payload", 1000, 1, DataPattern::RANDOM, 4},
      {"Large payload", 1000, 64, DataPattern::RANDOM, 4},
      {"No payload", 1000, 0, DataPattern::RANDOM, 4},

      // Thread scalability verification
      {"Thread test (1)", 10000, 8, DataPattern::RANDOM, 1},
      {"Thread test (2)", 10000, 8, DataPattern::RANDOM, 2},
      {"Thread test (4)", 10000, 8, DataPattern::RANDOM, 4},
      {"Thread test (8)", 10000, 8, DataPattern::RANDOM, 8},
      {"Thread test (16)", 10000, 8, DataPattern::RANDOM, 16},

      // Special case and alignment testing
      {"All identical keys", 5000, 8, DataPattern::SORTED, 4},
      {"Power of 2 size", 4096, 8, DataPattern::RANDOM, 4},
      {"Prime size", 4099, 8, DataPattern::RANDOM, 4},
      {"Odd size", 12345, 8, DataPattern::RANDOM, 4},
  };

  size_t passed = 0;
  size_t total = test_cases.size();

  for (const auto &test : test_cases) {
    if (run_test_case(test)) {
      ++passed;
    }
  }

  std::cout << std::endl;
  std::cout << "=== Test Results ===" << std::endl;
  std::cout << "Passed: " << passed << "/" << total << std::endl;

  if (passed == total) {
    std::cout << "All tests PASSED! ✓" << std::endl;
    return 0;
  } else {
    std::cout << "Some tests FAILED! ✗" << std::endl;
    return 1;
  }
}
