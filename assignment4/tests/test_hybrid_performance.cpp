#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/hybrid/mpi_ff_mergesort.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declaration for sequential mergesort
void sequential_mergesort(std::vector<Record> &data);

/**
 * @brief Performance test configuration
 */
struct PerfTestConfig {
  size_t data_size;
  size_t payload_size;
  DataPattern pattern;
  size_t parallel_threads;
};

/**
 * @brief Performance results structure
 */
struct PerformanceResult {
  int mpi_processes;
  int parallel_threads;
  double total_time_ms;
  double speedup_vs_stdsort;
  double speedup_vs_sequential;
  double speedup_vs_1_node;
};

/**
 * @brief Runs single-node baseline sorts.
 *
 * This version generates fresh data for each test to avoid illegal copy
 * operations on the move-only Record type.
 */
void run_baselines(const PerfTestConfig &config, double &std_sort_time,
                   double &seq_sort_time) {
  // 1. Baseline: std::sort
  {
    auto temp_data_std =
        generate_data(config.data_size, config.payload_size, config.pattern);
    Timer timer_std;
    std::sort(temp_data_std.begin(), temp_data_std.end());
    std_sort_time = timer_std.elapsed_ms();
  }

  // 2. Baseline: sequential mergesort
  {
    auto temp_data_seq =
        generate_data(config.data_size, config.payload_size, config.pattern);
    Timer timer_seq;
    sequential_mergesort(temp_data_seq);
    seq_sort_time = timer_seq.elapsed_ms();
  }
}

/**
 * @brief Run hybrid MPI performance benchmark.
 */
PerformanceResult run_hybrid_benchmark(const PerfTestConfig &config, int rank,
                                       int mpi_world_size,
                                       double baseline_std_sort_ms,
                                       double baseline_seq_sort_ms,
                                       double baseline_1_node_ms) {
  std::vector<Record> data;
  if (rank == 0) {
    data = generate_data(config.data_size, config.payload_size, config.pattern);
  }

  hybrid::HybridConfig hybrid_config;
  hybrid_config.parallel_threads = config.parallel_threads;
  hybrid::HybridMergeSort sorter(hybrid_config);

  MPI_Barrier(MPI_COMM_WORLD);
  Timer timer;
  auto result = sorter.sort(data, config.payload_size);
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = timer.elapsed_ms();

  PerformanceResult perf_result = {};
  if (rank == 0) {
    perf_result.mpi_processes = mpi_world_size;
    perf_result.parallel_threads = config.parallel_threads;
    perf_result.total_time_ms = elapsed;
    perf_result.speedup_vs_stdsort =
        (elapsed > 0) ? baseline_std_sort_ms / elapsed : 0.0;
    perf_result.speedup_vs_sequential =
        (elapsed > 0) ? baseline_seq_sort_ms / elapsed : 0.0;
    perf_result.speedup_vs_1_node =
        (elapsed > 0) ? baseline_1_node_ms / elapsed : 0.0;
  }
  return perf_result;
}

void print_help(char *name) {
  std::cout << "Usage: " << name
            << " <threads> [data_size_M] [payload_B] [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --t-stdsort <ms>      Baseline time for std::sort\n";
  std::cout
      << "  --t-sequential <ms>   Baseline time for sequential mergesort\n";
  std::cout
      << "  --t-1node <ms>        Baseline time for hybrid sort on 1 node\n";
}

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "MPI does not support required threading level\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2) {
    if (rank == 0)
      print_help(argv[0]);
    MPI_Finalize();
    return 1;
  }

  PerfTestConfig config = {10000000, 16, DataPattern::RANDOM, 1};
  double t_stdsort = 0.0, t_sequential = 0.0, t_1node = 0.0;

  try {
    config.parallel_threads = std::stoul(argv[1]);
    if (argc > 2)
      config.data_size = std::stoul(argv[2]) * 1000000;
    if (argc > 3)
      config.payload_size = std::stoul(argv[3]);
    for (int i = 4; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--t-stdsort" && i + 1 < argc)
        t_stdsort = std::stod(argv[++i]);
      if (arg == "--t-sequential" && i + 1 < argc)
        t_sequential = std::stod(argv[++i]);
      if (arg == "--t-1node" && i + 1 < argc)
        t_1node = std::stod(argv[++i]);
    }
  } catch (const std::exception &e) {
    if (rank == 0)
      std::cerr << "Invalid argument: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double baseline_std_sort_ms = t_stdsort;
  double baseline_seq_sort_ms = t_sequential;
  double baseline_1_node_ms = t_1node;

  if (size == 1) {
    double temp_hybrid_time = 0;
    if (rank == 0) {
      run_baselines(config, baseline_std_sort_ms, baseline_seq_sort_ms);
      // We need the 1-node time to calculate its own speedup, so run it once.
      temp_hybrid_time =
          run_hybrid_benchmark(config, rank, size, 0, 0, 0).total_time_ms;
      baseline_1_node_ms = temp_hybrid_time;
    }
    // Broadcast the calculated baselines to all (even though it's just one
    // process).
    MPI_Bcast(&baseline_std_sort_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&baseline_seq_sort_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&baseline_1_node_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Create the final result struct.
    PerformanceResult res = {};
    if (rank == 0) {
      res.mpi_processes = 1;
      res.parallel_threads = config.parallel_threads;
      res.total_time_ms = baseline_1_node_ms;
      res.speedup_vs_stdsort = (res.total_time_ms > 0)
                                   ? baseline_std_sort_ms / res.total_time_ms
                                   : 0.0;
      res.speedup_vs_sequential = (res.total_time_ms > 0)
                                      ? baseline_seq_sort_ms / res.total_time_ms
                                      : 0.0;
      res.speedup_vs_1_node = 1.0;

      // Output the final CSV-like line
      std::cout << res.mpi_processes << "," << res.parallel_threads << ","
                << res.total_time_ms << "," << baseline_std_sort_ms << ","
                << baseline_seq_sort_ms << "," << baseline_1_node_ms << "\n";
    }

  } else {
    // For multi-node runs, use the provided baselines
    PerformanceResult res =
        run_hybrid_benchmark(config, rank, size, baseline_std_sort_ms,
                             baseline_seq_sort_ms, baseline_1_node_ms);
    if (rank == 0) {
      std::cout << res.mpi_processes << "," << res.parallel_threads << ","
                << res.total_time_ms << "," << baseline_std_sort_ms << ","
                << baseline_seq_sort_ms << "," << baseline_1_node_ms << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}
