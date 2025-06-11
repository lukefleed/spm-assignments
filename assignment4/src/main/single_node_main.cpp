#include "../../include/csv_format.h"
#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

// Forward declaration for the parallel implementation
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

/**
 * @brief Validate sort operation correctness
 *
 * Verifies size preservation, key content preservation, and sort order.
 */
bool validate_result(const std::vector<Record> &sorted_data,
                     const std::vector<Record> &original_data) {
  // Check size preservation
  if (sorted_data.size() != original_data.size()) {
    std::cerr << "\n  [!] Validation Error: Size mismatch! Expected "
              << original_data.size() << ", but got " << sorted_data.size()
              << ".\n";
    return false;
  }

  // Check sort order
  if (!is_sorted(sorted_data)) {
    std::cerr << "\n  [!] Validation Error: Output is not sorted.\n";
    return false;
  }

  // Check key content preservation using multisets
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

/**
 * @brief Single-node mergesort comparison benchmark
 */
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  // CSV output setup
  std::ofstream csv_file;
  if (config.csv_output) {
    std::string filename = config.csv_filename;
    if (filename.empty()) {
      // Generate default filename with timestamp
      auto now = std::time(nullptr);
      auto tm = *std::localtime(&now);
      char buffer[80];
      std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
      filename = csv_format::generate_results_filename("single_node", buffer);
    }

    csv_file.open(filename);
    if (!csv_file.is_open()) {
      std::cerr << "Error: Cannot open CSV file " << filename << " for writing"
                << std::endl;
      return 1;
    }

    // Write CSV header
    csv_format::write_single_node_csv_header(csv_file);

    if (config.verbose) {
      std::cout << "CSV output will be written to: " << filename << std::endl;
    }
  }

  // Display benchmark configuration
  if (!config.csv_output || config.verbose) {
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
  }

  // Generate canonical dataset for all tests
  auto original_data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  // Setup results table
  if (!config.csv_output || config.verbose) {
    std::cout << std::left << std::setw(25) << "Implementation" << std::right
              << std::setw(15) << "Time (ms)" << std::setw(15) << "Speedup"
              << std::setw(15) << "Valid\n";
    std::cout << std::string(70, '-') << "\n";
  }

  double baseline_time = 0;
  double sequential_time = 0;

  // Benchmark 1: std::sort (baseline reference)
  {
    auto data = copy_records(original_data);
    Timer t;
    std::sort(data.begin(), data.end());
    double ms = t.elapsed_ms();
    baseline_time = ms; // Store for speedup calculations

    bool valid = config.validate ? validate_result(data, original_data) : true;

    if (!config.csv_output || config.verbose) {
      std::cout << std::left << std::setw(25) << "std::sort" << std::right
                << std::setw(15) << std::fixed << std::setprecision(2) << ms
                << std::setw(15) << "1.00x" << std::setw(15)
                << (valid ? "✓" : "✗") << "\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      double throughput =
          (config.array_size / 1000000.0) / (ms / 1000.0); // MRec/sec
      csv_format::write_single_node_csv_row(
          csv_file, "single_node", "std::sort", config.array_size,
          config.payload_size, 1, ms, throughput, 1.0, 0.0, 100.0, valid);
    }
  }

  // Benchmark 2: Sequential mergesort implementation
  {
    auto data = copy_records(original_data);
    Timer t;
    sequential_mergesort(data);
    double ms = t.elapsed_ms();
    sequential_time = ms; // Store for parallel speedup calculations

    bool valid = config.validate ? validate_result(data, original_data) : true;

    if (!config.csv_output || config.verbose) {
      std::cout << std::left << std::setw(25) << "Sequential MergeSort"
                << std::right << std::setw(15) << std::fixed
                << std::setprecision(2) << ms << std::setw(15) << std::fixed
                << std::setprecision(2) << baseline_time / ms << "x"
                << std::setw(15) << (valid ? "✓" : "✗") << "\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      double throughput =
          (config.array_size / 1000000.0) / (ms / 1000.0); // MRec/sec
      double speedup_vs_std = baseline_time / ms;
      csv_format::write_single_node_csv_row(
          csv_file, "single_node", "Sequential_MergeSort", config.array_size,
          config.payload_size, 1, ms, throughput, speedup_vs_std, 1.0, 100.0,
          valid);
    }
  }

  // Benchmark 3: FastFlow parallel mergesort
  {
    auto data = copy_records(original_data);
    Timer t;
    parallel_mergesort(data, config.num_threads);
    double ms = t.elapsed_ms();

    bool valid = config.validate ? validate_result(data, original_data) : true;

    if (!config.csv_output || config.verbose) {
      std::cout << std::left << std::setw(25) << "FF Parallel MergeSort"
                << std::right << std::setw(15) << std::fixed
                << std::setprecision(2) << ms << std::setw(15) << std::fixed
                << std::setprecision(2) << baseline_time / ms << "x"
                << std::setw(15) << (valid ? "✓" : "✗") << "\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      double throughput =
          (config.array_size / 1000000.0) / (ms / 1000.0); // MRec/sec
      double speedup_vs_std = baseline_time / ms;
      double speedup_vs_sequential = sequential_time / ms;
      double efficiency = (speedup_vs_sequential / config.num_threads) * 100.0;
      csv_format::write_single_node_csv_row(
          csv_file, "single_node", "FF_Parallel_MergeSort", config.array_size,
          config.payload_size, config.num_threads, ms, throughput,
          speedup_vs_std, speedup_vs_sequential, efficiency, valid);
    }
  }

  if (csv_file.is_open()) {
    csv_file.close();
  }

  return 0;
}
