/**
 * @file test_performance.cpp
 * @brief Simple performance benchmarking suite for single-node mergesort
 */

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
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads);

/**
 * @brief Performance test result structure
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
 * @brief Write CSV header
 */
void write_csv_header(std::ofstream &file) {
  file << "implementation,array_size,payload_size,num_threads,execution_time_"
          "ms,throughput_mb_per_sec,speedup\n";
}

/**
 * @brief Write test result to CSV
 */
void write_csv_row(std::ofstream &file, const TestResult &result) {
  file << result.implementation << "," << result.array_size << ","
       << result.payload_size << "," << result.num_threads << "," << std::fixed
       << std::setprecision(3) << result.execution_time_ms << ","
       << std::setprecision(2) << result.throughput_mb_per_sec << ","
       << std::setprecision(3) << result.speedup << "\n";
}

/**
 * @brief Run a single performance test
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

  // Run benchmark
  Timer timer;
  timer.start();

  if (implementation == "std::sort") {
    std::sort(data.begin(), data.end());
  } else if (implementation == "Sequential") {
    sequential_mergesort(data);
  } else if (implementation == "FastFlow") {
    ff_pipeline_two_farms_mergesort(data, num_threads);
  }

  result.execution_time_ms = timer.elapsed_ms();

  // Calculate throughput
  size_t data_size_bytes =
      (array_size *
       (sizeof(Record) - sizeof(std::unique_ptr<char[]>) + payload_size));
  double data_size_mb =
      static_cast<double>(data_size_bytes) / (1024.0 * 1024.0);
  result.throughput_mb_per_sec =
      data_size_mb / (result.execution_time_ms / 1000.0);

  // Calculate speedup
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
 * @brief Thread scaling test: 10M records, 1-12 threads
 */
void run_thread_scaling_test(std::ofstream &csv_file) {
  std::cout << "\n=== Thread Scaling Test (10M records, 64B payload) ==="
            << std::endl;

  const size_t array_size = 100000000; // 10M records
  const size_t payload_size = 64;      // 64 bytes payload

  std::vector<size_t> thread_counts = {2, 4, 6, 8, 10, 12};

  const int w_impl = 15;
  const int w_threads = 10;
  const int w_time = 15;
  const int w_speedup1 = 22;
  const int w_speedup2 = 25;
  const int w_tput = 20;

  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_threads) << "Threads" << std::setw(w_time)
            << "Time (ms)" << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_threads + w_time + w_speedup1 +
                               w_speedup2 + w_tput,
                           '-')
            << std::endl;

  auto std_sort_result =
      run_performance_test("std::sort", array_size, payload_size, 1);
  write_csv_row(csv_file, std_sort_result);

  std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_threads) << "1"
            << std::setw(w_time) << std::fixed << std::setprecision(1)
            << std_sort_result.execution_time_ms << std::setw(w_speedup1)
            << "1.00x" << std::setw(w_speedup2) << "-" << std::setw(w_tput)
            << std::setprecision(1) << std_sort_result.throughput_mb_per_sec
            << std::endl;

  auto sequential_result =
      run_performance_test("Sequential", array_size, payload_size, 1);
  write_csv_row(csv_file, sequential_result);

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

  for (size_t threads : thread_counts) {
    auto ff_result =
        run_performance_test("FastFlow", array_size, payload_size, threads);
    write_csv_row(csv_file, ff_result);

    double ff_vs_std =
        std_sort_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        sequential_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "FastFlow" << std::setw(w_threads)
              << threads << std::setw(w_time) << std::fixed
              << std::setprecision(1) << ff_result.execution_time_ms
              << std::setw(w_speedup1) << ff_vs_std_ss.str()
              << std::setw(w_speedup2) << ff_vs_seq_ss.str()
              << std::setw(w_tput) << std::setprecision(1)
              << ff_result.throughput_mb_per_sec << std::endl;
  }
}

/**
 * @brief Array size scaling test: 8 threads, 100K-10M records
 */
void run_array_size_test(std::ofstream &csv_file) {
  std::cout << "\n=== Array Size Scaling Test (8 threads, 64B payload) ==="
            << std::endl;

  const size_t payload_size = 64;
  const size_t num_threads = 8;

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

  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_size) << "Size" << std::setw(w_time) << "Time (ms)"
            << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_size + w_time + w_speedup1 + w_speedup2 +
                               w_tput,
                           '-')
            << std::endl;

  for (const auto &size_info : sizes) {
    size_t size = size_info.second;

    auto std_result = run_performance_test("std::sort", size, payload_size, 1);
    write_csv_row(csv_file, std_result);

    std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_size)
              << size_info.first << std::setw(w_time) << std::fixed
              << std::setprecision(1) << std_result.execution_time_ms
              << std::setw(w_speedup1) << "1.00x" << std::setw(w_speedup2)
              << "-" << std::setw(w_tput) << std::setprecision(1)
              << std_result.throughput_mb_per_sec << std::endl;

    auto seq_result = run_performance_test("Sequential", size, payload_size, 1);
    write_csv_row(csv_file, seq_result);

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

    auto ff_result =
        run_performance_test("FastFlow", size, payload_size, num_threads);
    write_csv_row(csv_file, ff_result);

    double ff_vs_std =
        std_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        seq_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "FastFlow" << std::setw(w_size)
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
 * @brief Payload size scaling test: 10M records, 8 threads
 */
void run_payload_size_test(std::ofstream &csv_file) {
  std::cout << "\n=== Payload Size Scaling Test (10M records, 8 threads) ==="
            << std::endl;

  const size_t array_size = 10000000;
  const size_t num_threads = 8;
  std::vector<size_t> payload_sizes = {8, 16, 32, 64, 128, 256};

  const int w_impl = 15;
  const int w_payload = 12;
  const int w_time = 15;
  const int w_speedup1 = 22;
  const int w_speedup2 = 25;
  const int w_tput = 20;

  std::cout << std::left << std::setw(w_impl) << "Implementation"
            << std::setw(w_payload) << "Payload (B)" << std::setw(w_time)
            << "Time (ms)" << std::setw(w_speedup1) << "Speedup (std::sort)"
            << std::setw(w_speedup2) << "Speedup (Sequential)"
            << std::setw(w_tput) << "Throughput (MB/s)" << std::endl;
  std::cout << std::string(w_impl + w_payload + w_time + w_speedup1 +
                               w_speedup2 + w_tput,
                           '-')
            << std::endl;

  for (size_t payload : payload_sizes) {
    auto std_result = run_performance_test("std::sort", array_size, payload, 1);
    write_csv_row(csv_file, std_result);

    std::cout << std::setw(w_impl) << "std::sort" << std::setw(w_payload)
              << payload << std::setw(w_time) << std::fixed
              << std::setprecision(1) << std_result.execution_time_ms
              << std::setw(w_speedup1) << "1.00x" << std::setw(w_speedup2)
              << "-" << std::setw(w_tput) << std::setprecision(1)
              << std_result.throughput_mb_per_sec << std::endl;

    auto seq_result =
        run_performance_test("Sequential", array_size, payload, 1);
    write_csv_row(csv_file, seq_result);

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

    auto ff_result =
        run_performance_test("FastFlow", array_size, payload, num_threads);
    write_csv_row(csv_file, ff_result);

    double ff_vs_std =
        std_result.execution_time_ms / ff_result.execution_time_ms;
    double ff_vs_seq =
        seq_result.execution_time_ms / ff_result.execution_time_ms;

    std::stringstream ff_vs_std_ss, ff_vs_seq_ss;
    ff_vs_std_ss << std::fixed << std::setprecision(2) << ff_vs_std << "x";
    ff_vs_seq_ss << std::fixed << std::setprecision(2) << ff_vs_seq << "x";

    std::cout << std::setw(w_impl) << "FastFlow" << std::setw(w_payload)
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

int main() {
  std::cout << "Single Node Performance Benchmarking Suite" << std::endl;
  std::cout << "===========================================" << std::endl;

  std::ofstream csv_file("performance_results.csv");

  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot create performance_results.csv" << std::endl;
    return 1;
  }

  write_csv_header(csv_file);

  run_thread_scaling_test(csv_file);
  run_array_size_test(csv_file);
  run_payload_size_test(csv_file); // <-- Chiamata alla nuova funzione

  csv_file.close();

  std::cout << "\n=== Performance Testing Complete ===" << std::endl;
  std::cout << "Results saved to: performance_results.csv" << std::endl;

  return 0;
}
