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
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <vector>

/**
 * @brief Performance test configuration
 */
struct PerfTestConfig {
  size_t data_size;
  size_t payload_size;
  DataPattern pattern;
  size_t ff_threads;
  std::string description;
  size_t iterations;
};

/**
 * @brief Global baseline time storage for speedup calculation across process
 * counts
 */
static double g_baseline_time = 0.0;

/**
 * @brief Run performance benchmark for hybrid implementation
 */
void run_hybrid_benchmark(const PerfTestConfig &config, int rank, int size) {
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
      std::cout << "ERROR: Result not sorted!\n";
      return;
    }

    // Store baseline for single process case
    if (size == 1) {
      g_baseline_time = elapsed;
    }

    // Calculate throughput (records per second)
    double throughput = config.data_size / (elapsed / 1000.0);

    std::cout << "MPI Processes: " << size << "\n";
    std::cout << "Execution Time: " << std::fixed << std::setprecision(2)
              << elapsed << " ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(0)
              << throughput << " records/sec\n";

    // Calculate and display speedup
    if (g_baseline_time > 0.0 && size > 1) {
      double speedup = g_baseline_time / elapsed;
      double efficiency = speedup / size;
      std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup
                << "x\n";
      std::cout << "Efficiency: " << std::fixed << std::setprecision(1)
                << (efficiency * 100) << "%\n";
    } else if (size == 1) {
      std::cout << "Baseline case\n";
    }
    std::cout << std::string(40, '-') << "\n";
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

  if (rank == 0) {
    std::cout
        << "Hybrid MPI+FastFlow Mergesort - Scientific Performance Analysis\n";
    std::cout
        << "==============================================================\n";
    std::cout << "Fixed Configuration: 10M records, 64B payload, 4 FF "
                 "threads/process\n";
    std::cout << "Data Pattern: Random (unsorted)\n\n";
  }

  // Single fixed configuration for scientific analysis
  PerfTestConfig config = {10000000,      64, DataPattern::RANDOM, 4,
                           "Random Data", 1};

  try {
    run_hybrid_benchmark(config, rank, size);
  } catch (const std::exception &e) {
    if (rank == 0) {
      std::cout << "Error during benchmark: " << e.what() << "\n";
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
