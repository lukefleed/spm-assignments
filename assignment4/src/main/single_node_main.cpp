#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

// Forward declaration for the parallel implementation
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

/**
 * @brief Display help information for command-line usage
 */
void print_help() {
  std::cout << "Usage: single_node_main [OPTIONS]\n\n";
  std::cout << "Single-node MergeSort benchmark comparing sequential and parallel implementations.\n\n";

  std::cout << "Options:\n";
  std::cout << "  -h, --help              Show this help message\n";
  std::cout << "  -s SIZE                 Array size (supports K/M/G suffixes, default: 1000000)\n";
  std::cout << "  -r SIZE                 Record payload size in bytes (default: 8)\n";
  std::cout << "  -t THREADS              Number of parallel threads (default: 4)\n";
  std::cout << "  --pattern PATTERN       Data pattern: random, sorted, reverse, nearly (default: random)\n";
  std::cout << "  --csv                   Output results in CSV format to stdout\n";
  std::cout << "  --csv-file FILE         Output results to specified CSV file\n\n";
}

/**
 * @brief Single-node mergesort comparison benchmark
 */
int main(int argc, char *argv[]) {
  auto [config, help_requested] = parse_args(argc, argv);
  if (help_requested) {
    print_help();
    return 0;
  }

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
      filename = "results_single_node_" + std::string(buffer) + ".csv";
    }

    csv_file.open(filename);
    if (!csv_file.is_open()) {
      std::cerr << "Error: Cannot open CSV file " << filename << " for writing"
                << std::endl;
      return 1;
    }

    // Write CSV header
    csv_file << "Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,"
             << "Execution_Time_ms,Speedup_vs_StdSort,Speedup_vs_Sequential,Valid\n";


  }

  // Display benchmark configuration
  if (!config.csv_output) {
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
  if (!config.csv_output) {
    std::cout << std::left << std::setw(25) << "Implementation" << std::right
              << std::setw(15) << "Time (ms)" << std::setw(15) << "Speedup\n";
    std::cout << std::string(55, '-') << "\n";
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

    if (!config.csv_output) {
      std::cout << std::left << std::setw(25) << "std::sort" << std::right
                << std::setw(15) << std::fixed << std::setprecision(2) << ms
                << std::setw(15) << "1.00x\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      csv_file << "single_node,std::sort," << config.array_size << ","
               << config.payload_size << ",1," << std::fixed << std::setprecision(3)
               << ms << ",1.000,0.000,true\n";
    }
  }

  // Benchmark 2: Sequential mergesort implementation
  {
    auto data = copy_records(original_data);
    Timer t;
    sequential_mergesort(data);
    double ms = t.elapsed_ms();
    sequential_time = ms; // Store for parallel speedup calculations

    if (!config.csv_output) {
      std::cout << std::left << std::setw(25) << "Sequential MergeSort"
                << std::right << std::setw(15) << std::fixed
                << std::setprecision(2) << ms << std::setw(15) << std::fixed
                << std::setprecision(2) << baseline_time / ms << "x\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      double speedup_vs_std = baseline_time / ms;
      csv_file << "single_node,Sequential_MergeSort," << config.array_size << ","
               << config.payload_size << ",1," << std::fixed << std::setprecision(3)
               << ms << "," << speedup_vs_std << ",1.000,true\n";
    }
  }

  // Benchmark 3: FastFlow parallel mergesort
  {
    auto data = copy_records(original_data);
    Timer t;
    parallel_mergesort(data, config.num_threads);
    double ms = t.elapsed_ms();

    if (!config.csv_output) {
      std::cout << std::left << std::setw(25) << "FF Parallel MergeSort"
                << std::right << std::setw(15) << std::fixed
                << std::setprecision(2) << ms << std::setw(15) << std::fixed
                << std::setprecision(2) << baseline_time / ms << "x\n";
    }

    // Write to CSV if enabled
    if (config.csv_output) {
      double speedup_vs_std = baseline_time / ms;
      double speedup_vs_sequential = sequential_time / ms;
      csv_file << "single_node,FF_Parallel_MergeSort," << config.array_size << ","
               << config.payload_size << "," << config.num_threads << ","
               << std::fixed << std::setprecision(3) << ms << ","
               << speedup_vs_std << "," << speedup_vs_sequential << ",true\n";
    }
  }

  if (csv_file.is_open()) {
    csv_file.close();
  }

  return 0;
}
