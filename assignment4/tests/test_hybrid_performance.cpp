/**
 * @file test_hybrid_performance.cpp
 * @brief Performance benchmarking suite for hybrid MPI+FastFlow mergesort
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
  size_t ff_threads; // This will be set from command line
  // std::string description; // No longer used for output from C++
  size_t iterations;
};

/**
 * @brief Run performance benchmark for hybrid implementation
 * @param config The performance test configuration.
 * @param rank The MPI rank of the current process.
 * @param mpi_world_size The total number of MPI processes.
 * @param baseline_file_path Path to the file for storing/reading baseline time.
 */
void run_hybrid_benchmark(const PerfTestConfig &config, int rank,
                          int mpi_world_size,
                          const std::string &baseline_file_path) {
  MPI_Barrier(MPI_COMM_WORLD);

  // Generate test data
  auto data =
      generate_data(config.data_size, config.payload_size, config.pattern);

  // Setup hybrid sorter
  hybrid::HybridConfig hybrid_config;
  hybrid_config.ff_threads = config.ff_threads;
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

    double baseline_time_for_calc = 0.0;

    if (mpi_world_size == 1) {
      baseline_time_for_calc = elapsed;
      std::ofstream baseline_out_file(baseline_file_path);
      if (baseline_out_file.is_open()) {
        baseline_out_file << std::fixed << std::setprecision(10) << elapsed;
        baseline_out_file.close();
      } else {
        std::cerr << "Error: Could not write to baseline file: "
                  << baseline_file_path << std::endl;
        // Proceed without saving baseline, speedup will be N/A for others
      }
    } else {
      std::ifstream baseline_in_file(baseline_file_path);
      if (baseline_in_file.is_open()) {
        baseline_in_file >> baseline_time_for_calc;
        baseline_in_file.close();
        if (baseline_time_for_calc <= 0.0) {
          std::cerr << "Warning: Invalid baseline time read from "
                    << baseline_file_path << std::endl;
          baseline_time_for_calc = 0.0; // Mark as unavailable
        }
      } else {
        std::cerr << "Warning: Could not read baseline file: "
                  << baseline_file_path << ". Speedup/Efficiency will be N/A."
                  << std::endl;
        // baseline_time_for_calc remains 0.0
      }
    }

    // Calculate throughput (MRec/s)
    double throughput_mrecs =
        (static_cast<double>(config.data_size) / 1000000.0) /
        (elapsed / 1000.0);

    // Print formatted row
    // MPI Procs   Time (ms)      Throughput (MRec/s)   Speedup      Efficiency
    // (%)
    // ----------- -------------- ------------------- ------------
    // ---------------
    std::cout << std::left << std::setw(11) << mpi_world_size << std::right
              << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
              << std::right << std::setw(19) << std::fixed
              << std::setprecision(2) << throughput_mrecs;

    if (mpi_world_size == 1) {
      std::cout << std::right << std::setw(12) << std::fixed
                << std::setprecision(2) << 1.00 << std::right << std::setw(15)
                << std::fixed << std::setprecision(1) << 100.0;
    } else if (baseline_time_for_calc > 0.0) {
      double speedup = baseline_time_for_calc / elapsed;
      double efficiency = speedup / mpi_world_size;
      std::cout << std::right << std::setw(12) << std::fixed
                << std::setprecision(2) << speedup << std::right
                << std::setw(15) << std::fixed << std::setprecision(1)
                << (efficiency * 100.0);
    } else {
      std::cout << std::right << std::setw(12) << "N/A" << std::right
                << std::setw(15) << "N/A";
    }
    std::cout << std::endl;
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

  if (argc < 3) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0] << " <ff_threads> <baseline_file_path>"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t ff_threads_arg;
  std::string baseline_file_path_arg;

  try {
    ff_threads_arg = std::stoul(argv[1]);
    baseline_file_path_arg = argv[2];
  } catch (const std::invalid_argument &ia) {
    if (rank == 0) {
      std::cerr << "Invalid argument: " << ia.what() << std::endl;
      std::cerr << "Usage: " << argv[0] << " <ff_threads> <baseline_file_path>"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  } catch (const std::out_of_range &oor) {
    if (rank == 0) {
      std::cerr << "Argument out of range: " << oor.what() << std::endl;
      std::cerr << "Usage: " << argv[0] << " <ff_threads> <baseline_file_path>"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Single fixed configuration for data size and pattern
  // ff_threads is now set from command line argument
  PerfTestConfig config = {10000000, 64, DataPattern::RANDOM, ff_threads_arg,
                           1};

  try {
    run_hybrid_benchmark(config, rank, size, baseline_file_path_arg);
  } catch (const std::exception &e) {
    if (rank == 0) {
      // Use std::cerr for errors to not interfere with table output to stdout
      std::cerr << "Error during benchmark: " << e.what() << "\n";
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
