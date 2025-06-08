/**
 * @file test_pevoid parallel_mergesort(std::vector<Record> &data, const size_t
 * num_threads);ormance.cpp
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
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

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
  } else if (implementation == "Parallel") {
    parallel_mergesort(data, num_threads);
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
 * @brief Thread scaling test with configurable parameters
 */
void run_thread_scaling_test(std::ofstream &csv_file,
                             const std::vector<size_t> &thread_counts,
                             size_t array_size, size_t payload_size) {
  std::cout << "\n=== Thread Scaling Test (" << array_size / 1000000
            << "M records, " << payload_size << "B payload) ===" << std::endl;

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
        run_performance_test("Parallel", array_size, payload_size, threads);
    write_csv_row(csv_file, ff_result);

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
        run_performance_test("Parallel", size, payload_size, num_threads);
    write_csv_row(csv_file, ff_result);

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
        run_performance_test("Parallel", array_size, payload, num_threads);
    write_csv_row(csv_file, ff_result);

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
 * @brief Parse command line arguments for flexible testing
 */
struct TestConfig {
  std::vector<size_t> thread_counts;
  size_t array_size;
  size_t payload_size;
  bool run_size_scaling;
  bool run_payload_scaling;
};

TestConfig parse_test_args(int argc, char *argv[]) {
  TestConfig config;

  // Default values
  config.thread_counts = {2, 4, 6, 8, 10, 12, 24};
  config.array_size = 10000000; // 10M
  config.payload_size = 64;
  config.run_size_scaling = false;
  config.run_payload_scaling = false;

  // Check for help first
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0]
          << " [THREAD_COUNTS] [ARRAY_SIZE_M] [PAYLOAD_SIZE_B] [OPTIONS]\n";
      std::cout << "\nParameters:\n";
      std::cout << "  THREAD_COUNTS    Space-separated list of thread counts "
                   "(e.g., \"2 4 8 12\")\n";
      std::cout << "  ARRAY_SIZE_M     Array size in millions of records "
                   "(default: 10)\n";
      std::cout << "  PAYLOAD_SIZE_B   Payload size in bytes (default: 64)\n";
      std::cout << "\nOptions:\n";
      std::cout << "  --size-scaling   Enable array size scaling test\n";
      std::cout << "  --payload-scaling Enable payload size scaling test\n";
      std::cout << "  --help, -h       Show this help message\n";
      std::cout << "\nExamples:\n";
      std::cout << "  " << argv[0] << " \"1 2 4 8\" 10 64\n";
      std::cout << "  " << argv[0] << " \"2 4 8\" 5 32 --size-scaling\n";
      std::cout << "  " << argv[0]
                << " \"1 2 4 8 12\" 10 64 --size-scaling --payload-scaling\n";
      exit(0);
    }
  }

  if (argc >= 2) {
    // Parse thread counts from first argument (space-separated string)
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

  if (argc >= 3) {
    try {
      // Parse array size in millions
      config.array_size = std::stoul(argv[2]) * 1000000;
    } catch (const std::exception &e) {
      std::cerr << "Error parsing array size: " << argv[2] << std::endl;
      std::cerr << "Expected a number in millions, e.g., 10" << std::endl;
      exit(1);
    }
  }

  if (argc >= 4) {
    try {
      // Parse payload size in bytes
      config.payload_size = std::stoul(argv[3]);
    } catch (const std::exception &e) {
      std::cerr << "Error parsing payload size: " << argv[3] << std::endl;
      std::cerr << "Expected a number in bytes, e.g., 64" << std::endl;
      exit(1);
    }
  }

  // Check for optional flags
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--size-scaling") {
      config.run_size_scaling = true;
    } else if (arg == "--payload-scaling") {
      config.run_payload_scaling = true;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      std::cerr << "Use --help for usage information" << std::endl;
      exit(1);
    }
  }

  return config;
}

int main(int argc, char *argv[]) {
  std::cout << "Single Node Performance Benchmarking Suite" << std::endl;
  std::cout << "===========================================" << std::endl;

  TestConfig config = parse_test_args(argc, argv);

  std::ofstream csv_file("performance_results.csv");

  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot create performance_results.csv" << std::endl;
    return 1;
  }

  write_csv_header(csv_file);

  // Always run thread scaling test with provided/default parameters
  run_thread_scaling_test(csv_file, config.thread_counts, config.array_size,
                          config.payload_size);

  // Optional tests
  if (config.run_size_scaling) {
    run_array_size_test(csv_file);
  }

  if (config.run_payload_scaling) {
    run_payload_size_test(csv_file);
  }

  csv_file.close();

  std::cout << "\n=== Performance Testing Complete ===" << std::endl;
  std::cout << "Results saved to: performance_results.csv" << std::endl;

  return 0;
}
