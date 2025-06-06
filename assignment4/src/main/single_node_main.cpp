#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>

// Forward declaration for the FastFlow implementation
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads);

/**
 * @brief Validates the result of a sort operation.
 *
 * Checks three conditions:
 * 1. The size of the sorted vector matches the original size.
 * 2. The multiset of keys in the sorted vector matches the original.
 * 3. The sorted vector is actually in non-decreasing order.
 *
 * @param sorted_data The vector after sorting.
 * @param original_data The vector before sorting.
 * @return True if all validation checks pass, false otherwise.
 */
bool validate_result(const std::vector<Record> &sorted_data,
                     const std::vector<Record> &original_data) {
  if (sorted_data.size() != original_data.size()) {
    std::cerr << "\n  [!] Validation Error: Size mismatch! Expected "
              << original_data.size() << ", but got " << sorted_data.size()
              << ".\n";
    return false;
  }

  if (!is_sorted(sorted_data)) {
    std::cerr << "\n  [!] Validation Error: Output is not sorted.\n";
    return false;
  }

  // Use multisets to verify that all original keys are present in the result
  std::multiset<unsigned long> original_keys;
  for (const auto &rec : original_data) {
    original_keys.insert(rec.key);
  }
  std::multiset<unsigned long> result_keys;
  for (const auto &rec : sorted_data) {
    result_keys.insert(rec.key);
  }

  if (original_keys != result_keys) {
    std::cerr
        << "\n  [!] Validation Error: Key content mismatch after sorting.\n";
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "=== Single Node MergeSort Comparison ===\n";
  std::cout << "Array size: " << config.array_size << " elements\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Total data: "
            << format_bytes(config.array_size *
                            (sizeof(Record::key) + config.payload_size))
            << "\n";
  std::cout << "Threads: " << config.num_threads << "\n";
  std::cout << "Pattern: ";
  switch (config.pattern) {
  case DataPattern::RANDOM:
    std::cout << "Random\n";
    break;
  case DataPattern::SORTED:
    std::cout << "Already Sorted\n";
    break;
  case DataPattern::REVERSE_SORTED:
    std::cout << "Reverse Sorted\n";
    break;
  case DataPattern::NEARLY_SORTED:
    std::cout << "Nearly Sorted\n";
    break;
  }
  std::cout << "\n";

  // Generate a single, canonical dataset for all tests to use
  auto original_data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  // Results table
  std::cout << std::left << std::setw(25) << "Implementation" << std::right
            << std::setw(15) << "Time (ms)" << std::setw(15) << "Speedup"
            << std::setw(15) << "Valid\n";
  std::cout << std::string(70, '-') << "\n";

  double baseline_time = 0;

  // Test std::sort (baseline)
  {
    auto data = copy_records(original_data);
    Timer t;
    std::sort(data.begin(), data.end());
    double ms = t.elapsed_ms();
    baseline_time = ms;

    bool valid = config.validate ? validate_result(data, original_data) : true;
    std::cout << std::left << std::setw(25) << "std::sort" << std::right
              << std::setw(15) << std::fixed << std::setprecision(2) << ms
              << std::setw(15) << "1.00x" << std::setw(15)
              << (valid ? "✓" : "✗") << "\n";
  }

  // Test sequential mergesort
  {
    auto data = copy_records(original_data);
    Timer t;
    sequential_mergesort(data);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? validate_result(data, original_data) : true;
    std::cout << std::left << std::setw(25) << "Sequential MergeSort"
              << std::right << std::setw(15) << std::fixed
              << std::setprecision(2) << ms << std::setw(15) << std::fixed
              << std::setprecision(2) << baseline_time / ms << "x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  // Test FastFlow pipeline with two farms
  {
    auto data = copy_records(original_data);
    Timer t;
    ff_pipeline_two_farms_mergesort(data, config.num_threads);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? validate_result(data, original_data) : true;
    std::cout << std::left << std::setw(25) << "FF Pipeline Two Farms"
              << std::right << std::setw(15) << std::fixed
              << std::setprecision(2) << ms << std::setw(15) << std::fixed
              << std::setprecision(2) << baseline_time / ms << "x"
              << std::setw(15) << (valid ? "✓" : "✗") << "\n";
  }

  return 0;
}
