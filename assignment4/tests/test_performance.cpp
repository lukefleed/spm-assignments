#include "../include/csv_format.h"
#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/fastflow/ff_mergesort.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Forward declaration for FastFlow implementation
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

/**
 * @brief Performance metrics container for benchmark results
 */
struct TestResult {
  std::string implementation;
  size_t array_size;
  size_t payload_size;
  size_t num_threads;
  double execution_time_ms;
  double throughput_mb_per_sec;
  double speedup;
};

/**
 * @brief Convert TestResult to standardized CSV format and write to file
 */
void write_standardized_csv_row(std::ofstream &file, const TestResult &result,
                                double sequential_time_ms) {
  // Calculate metrics for standardized format
  double throughput_mrec_per_sec =
      (static_cast<double>(result.array_size) / 1000000.0) /
      (result.execution_time_ms / 1000.0);

  // For efficiency calculation (only meaningful for parallel implementations)
  double efficiency = result.num_threads > 1
                          ? (result.speedup / result.num_threads) * 100.0
                          : 100.0;

  // Calculate speedup vs sequential
  double speedup_vs_sequential =
      sequential_time_ms > 0.0 ? sequential_time_ms / result.execution_time_ms
                               : 1.0;

  csv_format::write_single_node_csv_row(
      file, "performance_test", result.implementation, result.array_size,
      result.payload_size, result.num_threads, result.execution_time_ms,
      throughput_mrec_per_sec, result.speedup, speedup_vs_sequential,
      efficiency, true);
}

/**
 * @brief Execute and measure performance of sorting implementation
 * @param implementation Algorithm name
 * @param array_size Number of records to sort
 * @param payload_size Record payload size in bytes
 * @param num_threads Thread count for parallel implementations
 * @param baseline_time_ms Baseline time for speedup calculation
 * @return TestResult with timing and throughput metrics
 */
TestResult run_performance_test(const std::string &implementation,
                                size_t array_size, size_t payload_size,
                                size_t num_threads,
                                double baseline_time_ms = 0.0) {

  TestResult result;
  result.implementation = implementation;
  result.array_size = array_size;
  result.payload_size = payload_size;
  result.num_threads = num_threads;

  // Generate test data
  auto data = generate_data(array_size, payload_size, DataPattern::RANDOM);

  // Execute timed benchmark
  Timer timer;
  timer.start();

  if (implementation == "std::sort") {
    std::sort(data.begin(), data.end());
  } else if (implementation == "Sequential") {
    sequential_mergesort(data);
  } else if (implementation == "Parallel") {
    parallel_mergesort(data, num_threads);
  }

  result.execution_time_ms = timer.elapsed_ms();

  // Calculate throughput accounting for actual record size
  size_t data_size_bytes =
      (array_size *
       (sizeof(Record) - sizeof(std::unique_ptr<char[]>) + payload_size));
  double data_size_mb =
      static_cast<double>(data_size_bytes) / (1024.0 * 1024.0);
  result.throughput_mb_per_sec =
      data_size_mb / (result.execution_time_ms / 1000.0);

  // Calculate speedup relative to baseline
  if (baseline_time_ms > 0.0) {
    result.speedup = baseline_time_ms / result.execution_time_ms;
  } else {
    result.speedup = 1.0;
  }

  // Verify correctness
  if (!is_sorted(data)) {
    std::cerr << "ERROR: " << implementation << " produced unsorted output!"
              << std::endl;
  }

  return result;
}

/**
 * @brief Test parallel scaling across multiple thread counts
 */
void run_thread_scaling_test(std::ofstream &csv_file,
                             const std::vector<size_t> &thread_counts,
                             size_t array_size, size_t payload_size) {
  std::cout << "\n=== Thread Scaling Test (" << array_size / 1000000
            << "M records, " << payload_size << "B payload) ===" << std::endl;

  // Table column widths
  const int w_impl = 15;
  const int w_threads = 10;
  const int w_time = 15;
  const int w_speedup1 = 22;
  const int w_speedup2 = 25;
  const int w_tput = 20;

  // Display table header
  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_threads) << "Threads" << std::setw(w_time)
            << "Time (ms)" << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_threads + w_time + w_speedup1 +
                               w_speedup2 + w_tput,
                           '-')
            << std::endl;

  // Benchmark std::sort (baseline)
  auto std_sort_result =
      run_performance_test("std::sort", array_size, payload_size, 1);
  write_standardized_csv_row(csv_file, std_sort_result, 0.0);

  std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_threads) << "1"
            << std::setw(w_time) << std::fixed << std::setprecision(1)
            << std_sort_result.execution_time_ms << std::setw(w_speedup1)
            << "1.00x" << std::setw(w_speedup2) << "-" << std::setw(w_tput)
            << std::setprecision(1) << std_sort_result.throughput_mb_per_sec
            << std::endl;

  // Benchmark sequential mergesort
  auto sequential_result =
      run_performance_test("Sequential", array_size, payload_size, 1);
  write_standardized_csv_row(csv_file, sequential_result,
                             sequential_result.execution_time_ms);

  double seq_vs_std =
      std_sort_result.execution_time_ms / sequential_result.execution_time_ms;

  std::stringstream speedup_std_ss;
  speedup_std_ss << std::fixed << std::setprecision(2) << seq_vs_std << "x";

  std::cout << std::setw(w_impl) << "Sequential" << std::setw(w_threads) << "1"
            << std::setw(w_time) << std::fixed << std::setprecision(1)
            << sequential_result.execution_time_ms << std::setw(w_speedup1)
            << speedup_std_ss.str() << std::setw(w_speedup2) << "1.00x"
            << std::setw(w_tput) << std::setprecision(1)
            << sequential_result.throughput_mb_per_sec << std::endl;

  // Benchmark parallel mergesort with different thread counts
  for (size_t threads : thread_counts) {
    auto ff_result =
        run_performance_test("Parallel", array_size, payload_size, threads,
                             std_sort_result.execution_time_ms);
    write_standardized_csv_row(csv_file, ff_result,
                               sequential_result.execution_time_ms);

    double ff_vs_std =
        std_sort_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        sequential_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "Parallel" << std::setw(w_threads)
              << threads << std::setw(w_time) << std::fixed
              << std::setprecision(1) << ff_result.execution_time_ms
              << std::setw(w_speedup1) << ff_vs_std_ss.str()
              << std::setw(w_speedup2) << ff_vs_seq_ss.str()
              << std::setw(w_tput) << std::setprecision(1)
              << ff_result.throughput_mb_per_sec << std::endl;
  }
}

/**
 * @brief Test performance across different array sizes
 */
void run_array_size_test(std::ofstream &csv_file) {
  std::cout << "\n=== Array Size Scaling Test (8 threads, 64B payload) ==="
            << std::endl;

  const size_t payload_size = 64;
  const size_t num_threads = 8;

  // Test dataset sizes from 100K to 10M records
  std::vector<std::pair<std::string, size_t>> sizes = {{"100K", 100000},
                                                       {"500K", 500000},
                                                       {"1M", 1000000},
                                                       {"5M", 5000000},
                                                       {"10M", 10000000}};

  const int w_impl = 15;
  const int w_size = 12;
  const int w_time = 15;
  const int w_speedup1 = 22;
  const int w_speedup2 = 25;
  const int w_tput = 20;

  // Display table header
  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_size) << "Size" << std::setw(w_time) << "Time (ms)"
            << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_size + w_time + w_speedup1 + w_speedup2 +
                               w_tput,
                           '-')
            << std::endl;

  // Test each array size
  for (const auto &size_info : sizes) {
    size_t size = size_info.second;

    // std::sort baseline
    auto std_result = run_performance_test("std::sort", size, payload_size, 1);
    write_standardized_csv_row(csv_file, std_result, 0.0);

    std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_size)
              << size_info.first << std::setw(w_time) << std::fixed
              << std::setprecision(1) << std_result.execution_time_ms
              << std::setw(w_speedup1) << "1.00x" << std::setw(w_speedup2)
              << "-" << std::setw(w_tput) << std::setprecision(1)
              << std_result.throughput_mb_per_sec << std::endl;

    // Sequential mergesort
    auto seq_result = run_performance_test("Sequential", size, payload_size, 1);
    write_standardized_csv_row(csv_file, seq_result,
                               seq_result.execution_time_ms);

    double seq_vs_std =
        std_result.execution_time_ms / seq_result.execution_time_ms;

    std::stringstream seq_vs_std_ss;
    seq_vs_std_ss << std::fixed << std::setprecision(2) << seq_vs_std << "x";

    std::cout << std::setw(w_impl) << "Sequential" << std::setw(w_size)
              << size_info.first << std::setw(w_time) << std::fixed
              << std::setprecision(1) << seq_result.execution_time_ms
              << std::setw(w_speedup1) << seq_vs_std_ss.str()
              << std::setw(w_speedup2) << "1.00x" << std::setw(w_tput)
              << std::setprecision(1) << seq_result.throughput_mb_per_sec
              << std::endl;

    // Parallel mergesort
    auto ff_result =
        run_performance_test("Parallel", size, payload_size, num_threads,
                             std_result.execution_time_ms);
    write_standardized_csv_row(csv_file, ff_result,
                               seq_result.execution_time_ms);

    double ff_vs_std =
        std_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        seq_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "Parallel" << std::setw(w_size)
              << size_info.first << std::setw(w_time) << std::fixed
              << std::setprecision(1) << ff_result.execution_time_ms
              << std::setw(w_speedup1) << ff_vs_std_ss.str()
              << std::setw(w_speedup2) << ff_vs_seq_ss.str()
              << std::setw(w_tput) << std::setprecision(1)
              << ff_result.throughput_mb_per_sec << std::endl;

    if (size_info.first != "10M") {
      std::cout << std::string(w_impl + w_size + w_time + w_speedup1 +
                                   w_speedup2 + w_tput,
                               '-')
                << std::endl;
    }
  }
}

/**
 * @brief Test performance across different payload sizes
 */
void run_payload_size_test(std::ofstream &csv_file) {
  std::cout << "\n=== Payload Size Scaling Test (10M records, 8 threads) ==="
            << std::endl;

  const size_t array_size = 10000000;
  const size_t num_threads = 8;
  // Test payload sizes from 8B to 256B
  std::vector<size_t> payload_sizes = {8, 16, 32, 64, 128, 256};

  const int w_impl = 15;
  const int w_payload = 12;
  const int w_time = 15;
  const int w_speedup1 = 22;
  const int w_speedup2 = 25;
  const int w_tput = 20;

  // Display table header
  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_payload) << "Payload (B)" << std::setw(w_time)
            << "Time (ms)" << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_payload + w_time + w_speedup1 +
                               w_speedup2 + w_tput,
                           '-')
            << std::endl;

  // Test each payload size
  for (size_t payload : payload_sizes) {
    // std::sort baseline
    auto std_result = run_performance_test("std::sort", array_size, payload, 1);
    write_standardized_csv_row(csv_file, std_result, 0.0);

    std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_payload)
              << payload << std::setw(w_time) << std::fixed
              << std::setprecision(1) << std_result.execution_time_ms
              << std::setw(w_speedup1) << "1.00x" << std::setw(w_speedup2)
              << "-" << std::setw(w_tput) << std::setprecision(1)
              << std_result.throughput_mb_per_sec << std::endl;

    // Sequential mergesort
    auto seq_result =
        run_performance_test("Sequential", array_size, payload, 1);
    write_standardized_csv_row(csv_file, seq_result,
                               seq_result.execution_time_ms);

    double seq_vs_std =
        std_result.execution_time_ms / seq_result.execution_time_ms;
    std::stringstream seq_vs_std_ss;
    seq_vs_std_ss << std::fixed << std::setprecision(2) << seq_vs_std << "x";

    std::cout << std::setw(w_impl) << "Sequential" << std::setw(w_payload)
              << payload << std::setw(w_time) << std::fixed
              << std::setprecision(1) << seq_result.execution_time_ms
              << std::setw(w_speedup1) << seq_vs_std_ss.str()
              << std::setw(w_speedup2) << "1.00x" << std::setw(w_tput)
              << std::setprecision(1) << seq_result.throughput_mb_per_sec
              << std::endl;

    // Parallel mergesort
    auto ff_result =
        run_performance_test("Parallel", array_size, payload, num_threads,
                             std_result.execution_time_ms);
    write_standardized_csv_row(csv_file, ff_result,
                               seq_result.execution_time_ms);

    double ff_vs_std =
        std_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        seq_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "Parallel" << std::setw(w_payload)
              << payload << std::setw(w_time) << std::fixed
              << std::setprecision(1) << ff_result.execution_time_ms
              << std::setw(w_speedup1) << ff_vs_std_ss.str()
              << std::setw(w_speedup2) << ff_vs_seq_ss.str()
              << std::setw(w_tput) << std::setprecision(1)
              << ff_result.throughput_mb_per_sec << std::endl;

    if (payload != 256) {
      std::cout << std::string(w_impl + w_payload + w_time + w_speedup1 +
                                   w_speedup2 + w_tput,
                               '-')
                << std::endl;
    }
  }
}

/**
 * @brief Test configuration parameters
 */
struct TestConfig {
  std::vector<size_t> thread_counts;
  size_t array_size;
  size_t payload_size;
};

/**
 * @brief Parse command line arguments into test configuration
 */
TestConfig parse_test_args(int argc, char *argv[]) {
  TestConfig config;

  // Default configuration
  config.thread_counts = {2, 4, 6, 8, 10, 12, 24};
  config.array_size = 10000000; // 10M records
  config.payload_size = 64;     // 64B payload

  // Check for help
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [THREAD_COUNTS] [ARRAY_SIZE_M] [PAYLOAD_SIZE_B]\n";
      std::cout << "\nParameters:\n";
      std::cout << "  THREAD_COUNTS    Space-separated list of thread counts "
                   "(e.g., \"2 4 8 12\")\n";
      std::cout << "  ARRAY_SIZE_M     Array size in millions of records "
                   "(default: 10)\n";
      std::cout << "  PAYLOAD_SIZE_B   Payload size in bytes (default: 64)\n";
      std::cout << "\nExamples:\n";
      std::cout << "  " << argv[0] << " \"1 2 4 8\" 10 64\n";
      std::cout << "  " << argv[0] << " \"2 4 8\" 5 32\n";
      std::cout << "\nNote: For array size and payload scaling tests, use:\n";
      std::cout << "  make benchmark_array_scaling\n";
      std::cout << "  make benchmark_payload_scaling\n";
      exit(0);
    }
  }

  // Parse thread counts
  if (argc >= 2) {
    std::string thread_str = argv[1];
    std::istringstream iss(thread_str);
    std::string token;
    config.thread_counts.clear();
    try {
      while (iss >> token) {
        config.thread_counts.push_back(std::stoul(token));
      }
    } catch (const std::exception &e) {
      std::cerr << "Error parsing thread counts: " << thread_str << std::endl;
      std::cerr << "Expected space-separated numbers, e.g., \"2 4 8 12\""
                << std::endl;
      exit(1);
    }
  }

  // Parse array size
  if (argc >= 3) {
    try {
      config.array_size = std::stoul(argv[2]) * 1000000;
    } catch (const std::exception &e) {
      std::cerr << "Error parsing array size: " << argv[2] << std::endl;
      std::cerr << "Expected a number in millions, e.g., 10" << std::endl;
      exit(1);
    }
  }

  // Parse payload size
  if (argc >= 4) {
    try {
      config.payload_size = std::stoul(argv[3]);
    } catch (const std::exception &e) {
      std::cerr << "Error parsing payload size: " << argv[3] << std::endl;
      std::cerr << "Expected a number in bytes, e.g., 64" << std::endl;
      exit(1);
    }
  }

  return config;
}

/**
 * @brief Performance benchmarking main
 */
int main(int argc, char *argv[]) {
  std::cout << "Single Node Performance Benchmarking Suite" << std::endl;
  std::cout << "===========================================" << std::endl;

  TestConfig config = parse_test_args(argc, argv);

  std::ofstream csv_file("performance_results.csv");

  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot create performance_results.csv" << std::endl;
    return 1;
  }

  csv_format::write_single_node_csv_header(csv_file);

  // Execute core thread scaling benchmark
  run_thread_scaling_test(csv_file, config.thread_counts, config.array_size,
                          config.payload_size);

  csv_file.close();

  std::cout << "\n=== Performance Testing Complete ===" << std::endl;
  std::cout << "Results saved to: performance_results.csv" << std::endl;
  std::cout << "\nFor array size and payload scaling analysis, use:"
            << std::endl;
  std::cout << "  make benchmark_array_scaling" << std::endl;
  std::cout << "  make benchmark_payload_scaling" << std::endl;

  return 0;
}
