/**
 * @file test_hybrid_performance.cpp
 * @brief Performance benchmarking suite for hybrid MPI+parallel mergesort
 */

#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/hybrid/mpi_ff_mergesort.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <fstream> // Required for file operations
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <stdexcept> // Required for std::invalid_argument, std::out_of_range
#include <string>    // Required for std::string, std::stoul
#include <vector>

/**
 * @brief Performance test configuration
 */
struct PerfTestConfig {
  size_t data_size;
  size_t payload_size;
  DataPattern pattern;
  size_t parallel_threads; // Number of threads for parallel implementation
  size_t iterations;
};

/**
 * @brief Result structure for hybrid performance test data
 */
struct HybridTestResult {
  std::string test_name;
  size_t data_size;
  size_t payload_size;
  int mpi_processes;
  int parallel_threads;
  double total_time_ms;
  double throughput_mrec_per_sec;
  double speedup;
  double efficiency_percent;

  HybridTestResult(const std::string &name, size_t size, size_t payload,
                   int processes, int threads, double time, double throughput,
                   double speedup_val, double efficiency)
      : test_name(name), data_size(size), payload_size(payload),
        mpi_processes(processes), parallel_threads(threads),
        total_time_ms(time), throughput_mrec_per_sec(throughput),
        speedup(speedup_val), efficiency_percent(efficiency) {}
};

/**
 * @brief Write CSV header for hybrid performance results
 */
void write_hybrid_csv_header(std::ofstream &file) {
  file << "Test_Name,Data_Size,Payload_Size,MPI_Processes,Parallel_Threads,"
       << "Total_Time_ms,Throughput_MRec_per_sec,Speedup,Efficiency_Percent\n";
}

/**
 * @brief Write a single result row to CSV file
 */
void write_hybrid_csv_row(std::ofstream &file, const HybridTestResult &result) {
  file << result.test_name << "," << result.data_size << ","
       << result.payload_size << "," << result.mpi_processes << ","
       << result.parallel_threads << "," << result.total_time_ms << ","
       << result.throughput_mrec_per_sec << "," << result.speedup << ","
       << result.efficiency_percent << "\n";
}

/**
 * @brief Read baseline time from CSV file
 * @param csv_filename Path to the CSV file
 * @param parallel_threads Number of parallel threads to match
 * @param data_size Data size to match
 * @param payload_size Payload size to match
 * @return Baseline time in milliseconds, or 0.0 if not found
 */
double read_baseline_from_csv(const std::string &csv_filename,
                              int parallel_threads, size_t data_size,
                              size_t payload_size) {
  std::ifstream file(csv_filename);
  if (!file.is_open()) {
    return 0.0;
  }

  std::string line;
  std::getline(file, line); // Skip header

  double baseline_time = 0.0;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
      tokens.push_back(token);
    }

    if (tokens.size() >= 9) {
      // Parse:
      // Test_Name,Data_Size,Payload_Size,MPI_Processes,Parallel_Threads,Total_Time_ms,Throughput_MRec_per_sec,Speedup,Efficiency_Percent
      try {
        size_t csv_data_size = std::stoul(tokens[1]);
        size_t csv_payload_size = std::stoul(tokens[2]);
        int csv_mpi_processes = std::stoi(tokens[3]);
        int csv_parallel_threads = std::stoi(tokens[4]);
        double csv_time = std::stod(tokens[5]);

        if (csv_mpi_processes == 1 &&
            csv_parallel_threads == parallel_threads &&
            csv_data_size == data_size && csv_payload_size == payload_size) {
          baseline_time = csv_time;
        }
      } catch (const std::exception &) {
        // Skip malformed lines
        continue;
      }
    }
  }

  return baseline_time;
}

/**
 * @brief Run performance benchmark for hybrid implementation
 * @param config The performance test configuration.
 * @param rank The MPI rank of the current process.
 * @param mpi_world_size The total number of MPI processes.
 * @param baseline_time_ms Baseline time in milliseconds for speedup
 * calculation.
 * @param csv_file Pointer to CSV output file (nullptr if not saving to CSV).
 */
void run_hybrid_benchmark(const PerfTestConfig &config, int rank,
                          int mpi_world_size, double baseline_time_ms,
                          std::ofstream *csv_file = nullptr) {
  MPI_Barrier(MPI_COMM_WORLD);

  // Generate test data
  auto data =
      generate_data(config.data_size, config.payload_size, config.pattern);

  // Setup hybrid sorter
  hybrid::HybridConfig hybrid_config;
  hybrid_config.parallel_threads = config.parallel_threads;
  hybrid::HybridMergeSort sorter(hybrid_config);

  MPI_Barrier(MPI_COMM_WORLD);
  Timer timer;

  // Run the sort
  auto result = sorter.sort(data, config.payload_size);

  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = timer.elapsed_ms();

  if (rank == 0) {
    // Verify correctness
    bool sorted = std::is_sorted(
        result.begin(), result.end(),
        [](const Record &a, const Record &b) { return a.key < b.key; });
    if (!sorted) {
      // Still print to cerr for critical errors, but not to stdout table
      std::cerr << "ERROR: Result not sorted for MPI Processes: "
                << mpi_world_size << "!\n";
      // Output a line indicating failure or skip, to maintain table structure
      // if desired For now, we just don't print a success line.
      return;
    }

    // Calculate throughput (MRec/s)
    double throughput_mrecs =
        (static_cast<double>(config.data_size) / 1000000.0) /
        (elapsed / 1000.0);

    // Calculate speedup and efficiency
    double speedup = 1.0;
    double efficiency_percent = 100.0;

    if (mpi_world_size > 1 && baseline_time_ms > 0.0) {
      speedup = baseline_time_ms / elapsed;
      efficiency_percent = (speedup / mpi_world_size) * 100.0;
    }

    // Print formatted row to console
    std::cout << std::left << std::setw(11) << mpi_world_size << std::right
              << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
              << std::right << std::setw(19) << std::fixed
              << std::setprecision(2) << throughput_mrecs;

    if (mpi_world_size == 1) {
      std::cout << std::right << std::setw(12) << std::fixed
                << std::setprecision(2) << 1.00 << std::right << std::setw(15)
                << std::fixed << std::setprecision(1) << 100.0;
    } else if (baseline_time_ms > 0.0) {
      std::cout << std::right << std::setw(12) << std::fixed
                << std::setprecision(2) << speedup << std::right
                << std::setw(15) << std::fixed << std::setprecision(1)
                << efficiency_percent;
    } else {
      std::cout << std::right << std::setw(12) << "N/A" << std::right
                << std::setw(15) << "N/A";
    }
    std::cout << std::endl;

    // Save to CSV if file is provided
    if (csv_file && csv_file->is_open()) {
      HybridTestResult result(
          "Hybrid_MPI_Parallel", config.data_size, config.payload_size,
          mpi_world_size, static_cast<int>(config.parallel_threads), elapsed,
          throughput_mrecs, speedup, efficiency_percent);
      write_hybrid_csv_row(*csv_file, result);
    }
  }
}

/**
 * @brief Main performance testing function
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

  if (argc < 2) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0]
                << " <parallel_threads> [data_size_millions] [payload_size] "
                   "[csv_filename] [--quiet]"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t parallel_threads_arg;
  size_t data_size_millions = 10; // Default 10M
  size_t payload_size_bytes = 64; // Default 64B
  std::string csv_filename = "";  // Empty means no CSV
  bool quiet_mode = false;        // Suppress headers and info messages

  try {
    parallel_threads_arg = std::stoul(argv[1]);

    // Optional parameters
    if (argc > 2) {
      data_size_millions = std::stoul(argv[2]);
    }
    if (argc > 3) {
      payload_size_bytes = std::stoul(argv[3]);
    }
    if (argc > 4) {
      csv_filename = argv[4];
    }
    if (argc > 5 && std::string(argv[5]) == "--quiet") {
      quiet_mode = true;
    }
  } catch (const std::invalid_argument &ia) {
    if (rank == 0) {
      std::cerr << "Invalid argument: " << ia.what() << std::endl;
      std::cerr << "Usage: " << argv[0]
                << " <parallel_threads> [data_size_millions] [payload_size] "
                   "[csv_filename] [--quiet]"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  } catch (const std::out_of_range &oor) {
    if (rank == 0) {
      std::cerr << "Argument out of range: " << oor.what() << std::endl;
      std::cerr << "Usage: " << argv[0]
                << " <parallel_threads> [data_size_millions] [payload_size] "
                   "[csv_filename] [--quiet]"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Configuration with actual data size (convert millions to absolute count)
  PerfTestConfig config = {data_size_millions * 1000000, payload_size_bytes,
                           DataPattern::RANDOM, parallel_threads_arg, 1};

  // Setup CSV file if requested (only on rank 0)
  std::ofstream csv_file;
  std::ofstream *csv_ptr = nullptr;

  if (rank == 0 && !csv_filename.empty()) {
    // Check if this is the first MPI process count (create header) or
    // subsequent (append)
    bool file_exists = std::ifstream(csv_filename).good();

    csv_file.open(csv_filename, std::ios::app);
    if (csv_file.is_open()) {
      csv_ptr = &csv_file;
      if (!file_exists) {
        write_hybrid_csv_header(csv_file);
      }
    } else if (rank == 0) {
      std::cerr << "Warning: Could not open CSV file: " << csv_filename
                << std::endl;
    }
  }

  if (rank == 0 && !quiet_mode) {
    std::cout << "\n=== Hybrid MPI+Parallel Performance Test ===\n";
    std::cout << "Data Size: " << data_size_millions
              << "M records, Payload: " << payload_size_bytes
              << " bytes, FF Threads: " << parallel_threads_arg << "\n\n";

    if (size == 1) {
      std::cout << "Running baseline measurement with 1 MPI process...\n";
    }

    std::cout << std::left << std::setw(11) << "MPI Procs" << std::right
              << std::setw(14) << "Time (ms)" << std::right << std::setw(19)
              << "Throughput (MRec/s)" << std::right << std::setw(12)
              << "Speedup" << std::right << std::setw(15) << "Efficiency (%)"
              << std::endl;
    std::cout << std::string(71, '-') << std::endl;
  }

  try {
    if (size == 1) {
      // Single process - establish baseline and save to CSV
      run_hybrid_benchmark(config, rank, size, 0.0, csv_ptr);

    } else {
      // Multi-process - read baseline from CSV if available
      double baseline_time_ms = 0.0;

      if (rank == 0 && !csv_filename.empty()) {
        baseline_time_ms = read_baseline_from_csv(
            csv_filename, static_cast<int>(parallel_threads_arg),
            config.data_size, config.payload_size);

        if (baseline_time_ms > 0.0) {
          if (!quiet_mode) {
            std::cout << "Found baseline time: " << std::fixed
                      << std::setprecision(2) << baseline_time_ms
                      << " ms (from CSV)\n";
          }
        } else {
          if (!quiet_mode) {
            std::cout
                << "Warning: No baseline found in CSV. Speedup will be N/A.\n";
          }
        }
      }

      // Broadcast baseline time to all processes
      MPI_Bcast(&baseline_time_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Run the actual multi-process test
      run_hybrid_benchmark(config, rank, size, baseline_time_ms, csv_ptr);
    }
  } catch (const std::exception &e) {
    if (rank == 0) {
      // Use std::cerr for errors to not interfere with table output to stdout
      std::cerr << "Error during benchmark: " << e.what() << "\n";
    }
    MPI_Finalize();
    return 1;
  }

  if (csv_file.is_open()) {
    csv_file.close();
  }

  MPI_Finalize();
  return 0;
}
