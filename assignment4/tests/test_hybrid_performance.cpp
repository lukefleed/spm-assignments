/**
 * @file test_hybrid_performance.cpp
 * @brief Production performance benchmarking suite for hybrid MPI+parallel
 * mergesort
 *
 * Implements dual-baseline performance analysis to isolate MPI distribution
 * effects from local parallelization benefits. Maintains fixed FastFlow threads
 * per process to ensure pure MPI scaling measurement.
 */

#include "../src/common/record.hpp"
#include "../src/common/timer.hpp"
#include "../src/common/utils.hpp"
#include "../src/fastflow/ff_mergesort.hpp"
#include "../src/hybrid/mpi_ff_mergesort.hpp"
#include "../src/sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Performance test configuration with dual baseline support
 */
struct PerfTestConfig {
  size_t data_size;        ///< Total number of records to sort
  size_t payload_size;     ///< Size of each record's payload in bytes
  DataPattern pattern;     ///< Data distribution pattern
  size_t parallel_threads; ///< Fixed FastFlow worker threads per MPI process
  size_t iterations;       ///< Number of test iterations
};

/**
 * @brief Comprehensive performance metrics with dual baseline analysis
 */
struct EnhancedHybridResult {
  std::string test_name; ///< Test identifier
  size_t data_size;      ///< Number of records processed
  size_t payload_size;   ///< Record payload size in bytes
  int mpi_processes;     ///< MPI process count
  int parallel_threads;  ///< FastFlow worker thread count per process
  double total_time_ms;  ///< Total execution time in milliseconds
  double
      throughput_mrec_per_sec; ///< Throughput in millions of records per second
  double sequential_speedup;   ///< Speedup relative to pure sequential baseline
  double
      parallel_speedup; ///< Speedup relative to single-node parallel baseline
  double mpi_efficiency_percent;   ///< MPI distribution efficiency percentage
  double total_efficiency_percent; ///< Overall parallel efficiency percentage

  EnhancedHybridResult(const std::string &name, size_t size, size_t payload,
                       int processes, int threads, double time,
                       double throughput, double seq_speedup,
                       double par_speedup, double mpi_eff, double total_eff)
      : test_name(name), data_size(size), payload_size(payload),
        mpi_processes(processes), parallel_threads(threads),
        total_time_ms(time), throughput_mrec_per_sec(throughput),
        sequential_speedup(seq_speedup), parallel_speedup(par_speedup),
        mpi_efficiency_percent(mpi_eff), total_efficiency_percent(total_eff) {}
};

/**
 * @brief Write enhanced CSV header with dual baseline metrics
 */
void write_enhanced_csv_header(std::ofstream &file) {
  file << "Test_Name,Data_Size,Payload_Size,MPI_Processes,Parallel_Threads,"
       << "Total_Time_ms,Throughput_MRec_per_sec,Sequential_Speedup,Parallel_"
          "Speedup,"
       << "MPI_Efficiency_Percent,Total_Efficiency_Percent\n";
}

/**
 * @brief Write enhanced result row to CSV file
 */
void write_enhanced_csv_row(std::ofstream &file,
                            const EnhancedHybridResult &result) {
  file << result.test_name << "," << result.data_size << ","
       << result.payload_size << "," << result.mpi_processes << ","
       << result.parallel_threads << "," << result.total_time_ms << ","
       << result.throughput_mrec_per_sec << "," << result.sequential_speedup
       << "," << result.parallel_speedup << "," << result.mpi_efficiency_percent
       << "," << result.total_efficiency_percent << "\n";
}

/**
 * @brief Extract dual baseline times from CSV results
 * @param csv_filename Path to CSV file containing historical results
 * @param parallel_threads FastFlow thread count to match
 * @param data_size Record count to match
 * @param payload_size Payload size to match
 * @return Pair of (sequential_time, parallel_time) in milliseconds
 *
 * Searches for both sequential and single-process parallel baselines
 * with matching parameters for comprehensive speedup analysis.
 */
std::pair<double, double>
read_dual_baseline_from_csv(const std::string &csv_filename,
                            int parallel_threads, size_t data_size,
                            size_t payload_size) {
  std::ifstream file(csv_filename);
  if (!file.is_open()) {
    return {0.0, 0.0};
  }

  std::string line;
  std::getline(file, line); // Skip header

  double sequential_time = 0.0;
  double parallel_time = 0.0;

  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
      tokens.push_back(token);
    }

    if (tokens.size() >= 11) {
      try {
        size_t csv_data_size = std::stoul(tokens[1]);
        size_t csv_payload_size = std::stoul(tokens[2]);
        int csv_mpi_processes = std::stoi(tokens[3]);
        int csv_parallel_threads = std::stoi(tokens[4]);
        double csv_time = std::stod(tokens[5]);
        const std::string &test_name = tokens[0];

        if (csv_data_size == data_size && csv_payload_size == payload_size) {
          if (test_name == "Sequential_Baseline" && csv_mpi_processes == 1 &&
              csv_parallel_threads == 1) {
            sequential_time = csv_time;
          } else if (test_name == "Parallel_Baseline" &&
                     csv_mpi_processes == 1 &&
                     csv_parallel_threads == parallel_threads) {
            parallel_time = csv_time;
          }
        }
      } catch (const std::exception &) {
        continue;
      }
    }
  }

  return {sequential_time, parallel_time};
}

/**
 * @brief Execute pure sequential baseline measurement
 * @param config Test configuration parameters
 * @param csv_file Output file stream for results persistence
 * @return Sequential execution time in milliseconds
 *
 * Establishes sequential baseline using std::sort for absolute speedup
 * reference.
 */
double run_sequential_baseline(const PerfTestConfig &config,
                               std::ofstream *csv_file = nullptr) {
  auto data =
      generate_data(config.data_size, config.payload_size, config.pattern);

  Timer timer;
  std::sort(data.begin(), data.end(),
            [](const Record &a, const Record &b) { return a.key < b.key; });
  double elapsed = timer.elapsed_ms();

  // Calculate performance metrics
  double throughput_mrecs =
      (static_cast<double>(config.data_size) / 1000000.0) / (elapsed / 1000.0);

  std::cout << std::left << std::setw(11) << "Sequential" << std::right
            << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
            << std::right << std::setw(19) << std::fixed << std::setprecision(2)
            << throughput_mrecs << std::right << std::setw(12) << "1.00"
            << std::right << std::setw(12) << "1.00" << std::right
            << std::setw(15) << "100.0" << std::right << std::setw(15)
            << "100.0" << std::endl;

  if (csv_file && csv_file->is_open()) {
    EnhancedHybridResult result("Sequential_Baseline", config.data_size,
                                config.payload_size, 1, 1, elapsed,
                                throughput_mrecs, 1.0, 1.0, 100.0, 100.0);
    write_enhanced_csv_row(*csv_file, result);
  }

  return elapsed;
}

/**
 * @brief Execute pure FastFlow baseline measurement
 * @param config Test configuration parameters
 * @param sequential_time_ms Sequential baseline for speedup calculation
 * @param csv_file Output file stream for results persistence
 * @return Single-node parallel execution time in milliseconds
 *
 * Establishes single-node parallel baseline using direct FastFlow call
 * to avoid MPI infrastructure overhead and provide consistent baseline
 * measurements between single-node and hybrid test suites.
 */
double run_parallel_baseline(const PerfTestConfig &config,
                             double sequential_time_ms,
                             std::ofstream *csv_file = nullptr) {
  auto data =
      generate_data(config.data_size, config.payload_size, config.pattern);

  Timer timer;
  parallel_mergesort(data, config.parallel_threads);
  double elapsed = timer.elapsed_ms();

  // Verify correctness
  bool sorted = std::is_sorted(
      data.begin(), data.end(),
      [](const Record &a, const Record &b) { return a.key < b.key; });
  if (!sorted) {
    throw std::runtime_error("Parallel baseline sort verification failed");
  }

  // Calculate performance metrics
  double throughput_mrecs =
      (static_cast<double>(config.data_size) / 1000000.0) / (elapsed / 1000.0);
  double sequential_speedup = sequential_time_ms / elapsed;
  double total_efficiency =
      (sequential_speedup / config.parallel_threads) * 100.0;

  std::cout << std::left << std::setw(11) << "1" << std::right << std::setw(14)
            << std::fixed << std::setprecision(2) << elapsed << std::right
            << std::setw(19) << std::fixed << std::setprecision(2)
            << throughput_mrecs << std::right << std::setw(12) << std::fixed
            << std::setprecision(2) << sequential_speedup << std::right
            << std::setw(12) << "1.00" << std::right << std::setw(15) << "100.0"
            << std::right << std::setw(15) << std::fixed << std::setprecision(1)
            << total_efficiency << std::endl;

  if (csv_file && csv_file->is_open()) {
    EnhancedHybridResult result(
        "Parallel_Baseline", config.data_size, config.payload_size, 1,
        static_cast<int>(config.parallel_threads), elapsed, throughput_mrecs,
        sequential_speedup, 1.0, 100.0, total_efficiency);
    write_enhanced_csv_row(*csv_file, result);
  }

  return elapsed;
}
/**
 * @brief Execute enhanced hybrid MPI+FastFlow mergesort benchmark
 * @param config Test configuration parameters
 * @param rank Current MPI process rank
 * @param mpi_world_size Total MPI process count
 * @param sequential_time_ms Sequential baseline time
 * @param parallel_time_ms Single-node parallel baseline time
 * @param csv_file Output file stream for results persistence
 *
 * Performs distributed sorting with fixed FastFlow threads per process.
 * Calculates dual speedup metrics to isolate MPI distribution effects.
 */
void run_enhanced_hybrid_benchmark(const PerfTestConfig &config, int rank,
                                   int mpi_world_size,
                                   double sequential_time_ms,
                                   double parallel_time_ms,
                                   bool quiet_mode = false,
                                   std::ofstream *csv_file = nullptr) {
  MPI_Barrier(MPI_COMM_WORLD);

  auto data =
      generate_data(config.data_size, config.payload_size, config.pattern);

  hybrid::HybridConfig hybrid_config;
  hybrid_config.parallel_threads = config.parallel_threads;
  hybrid::HybridMergeSort sorter(hybrid_config);

  MPI_Barrier(MPI_COMM_WORLD);
  Timer timer;

  auto result = sorter.sort(data, config.payload_size);

  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = timer.elapsed_ms();

  if (rank == 0) {
    // Verify sort correctness
    bool sorted = std::is_sorted(
        result.begin(), result.end(),
        [](const Record &a, const Record &b) { return a.key < b.key; });
    if (!sorted) {
      std::cerr << "ERROR: Result not sorted for MPI Processes: "
                << mpi_world_size << "!\n";
      return;
    }

    // Calculate comprehensive performance metrics
    double throughput_mrecs =
        (static_cast<double>(config.data_size) / 1000000.0) /
        (elapsed / 1000.0);

    double sequential_speedup =
        (sequential_time_ms > 0.0) ? sequential_time_ms / elapsed : 1.0;
    double parallel_speedup =
        (parallel_time_ms > 0.0) ? parallel_time_ms / elapsed : 1.0;

    // MPI efficiency: speedup relative to single-node parallel divided by MPI
    // processes
    double mpi_efficiency = (parallel_speedup / mpi_world_size) * 100.0;

    // Total efficiency: speedup relative to sequential divided by total threads
    size_t total_threads = mpi_world_size * config.parallel_threads;
    double total_efficiency = (sequential_speedup / total_threads) * 100.0;

    // Display results - in quiet mode, show only the essential data row
    if (quiet_mode) {
      // Simple format for makefile processing: just the data row
      std::cout << std::left << std::setw(11) << mpi_world_size << std::right
                << std::setw(14) << std::fixed << std::setprecision(2)
                << elapsed << std::right << std::setw(19) << std::fixed
                << std::setprecision(2) << throughput_mrecs;

      if (sequential_time_ms > 0.0) {
        std::cout << std::right << std::setw(12) << std::fixed
                  << std::setprecision(2) << sequential_speedup;
      } else {
        std::cout << std::right << std::setw(12) << "N/A";
      }

      if (parallel_time_ms > 0.0) {
        std::cout << std::right << std::setw(12) << std::fixed
                  << std::setprecision(2) << parallel_speedup;
      } else {
        std::cout << std::right << std::setw(12) << "N/A";
      }

      std::cout << std::right << std::setw(15) << std::fixed
                << std::setprecision(1) << mpi_efficiency << std::right
                << std::setw(15) << std::fixed << std::setprecision(1)
                << total_efficiency << std::endl;
    } else {
      // Full format with all details
      std::cout << std::left << std::setw(11) << mpi_world_size << std::right
                << std::setw(14) << std::fixed << std::setprecision(2)
                << elapsed << std::right << std::setw(19) << std::fixed
                << std::setprecision(2) << throughput_mrecs;

      if (sequential_time_ms > 0.0) {
        std::cout << std::right << std::setw(12) << std::fixed
                  << std::setprecision(2) << sequential_speedup;
      } else {
        std::cout << std::right << std::setw(12) << "N/A";
      }

      if (parallel_time_ms > 0.0) {
        std::cout << std::right << std::setw(12) << std::fixed
                  << std::setprecision(2) << parallel_speedup;
      } else {
        std::cout << std::right << std::setw(12) << "N/A";
      }

      std::cout << std::right << std::setw(15) << std::fixed
                << std::setprecision(1) << mpi_efficiency << std::right
                << std::setw(15) << std::fixed << std::setprecision(1)
                << total_efficiency << std::endl;
    }

    // Persist enhanced results to CSV
    if (csv_file && csv_file->is_open()) {
      EnhancedHybridResult result(
          "Hybrid_MPI_Parallel", config.data_size, config.payload_size,
          mpi_world_size, static_cast<int>(config.parallel_threads), elapsed,
          throughput_mrecs, sequential_speedup, parallel_speedup,
          mpi_efficiency, total_efficiency);
      write_enhanced_csv_row(*csv_file, result);
    }
  }
}

/**
 * @brief Enhanced hybrid MPI+FastFlow mergesort performance benchmarking suite
 *
 * Implements dual-baseline performance analysis with production-grade metrics:
 * - Sequential baseline: Pure std::sort performance reference
 * - Parallel baseline: Single-node MPI+FastFlow performance reference
 * - MPI efficiency: Measures pure distribution scaling effectiveness
 * - Total efficiency: Measures overall parallelization effectiveness
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
                   "[csv_filename] [--quiet] [--skip-baselines]"
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t parallel_threads_arg;
  size_t data_size_millions = 10;
  size_t payload_size_bytes = 64;
  std::string csv_filename = "";
  bool quiet_mode = false;
  bool skip_baselines = false;

  try {
    parallel_threads_arg = std::stoul(argv[1]);

    if (argc > 2)
      data_size_millions = std::stoul(argv[2]);
    if (argc > 3)
      payload_size_bytes = std::stoul(argv[3]);
    if (argc > 4)
      csv_filename = argv[4];

    for (int i = 5; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--quiet")
        quiet_mode = true;
      else if (arg == "--skip-baselines")
        skip_baselines = true;
    }
  } catch (const std::exception &e) {
    if (rank == 0) {
      std::cerr << "Invalid argument: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  PerfTestConfig config = {data_size_millions * 1000000, payload_size_bytes,
                           DataPattern::RANDOM, parallel_threads_arg, 1};

  // Initialize enhanced CSV output
  std::ofstream csv_file;
  std::ofstream *csv_ptr = nullptr;

  if (rank == 0 && !csv_filename.empty()) {
    bool file_exists = std::ifstream(csv_filename).good();
    csv_file.open(csv_filename, std::ios::app);
    if (csv_file.is_open()) {
      csv_ptr = &csv_file;
      if (!file_exists) {
        write_enhanced_csv_header(csv_file);
      }
    }
  }

  if (rank == 0) {
    if (!quiet_mode) {
      std::cout << "\n=== Hybrid MPI+Parallel Performance Test ===\n";
      std::cout << "Data Size: " << data_size_millions
                << "M records, Payload: " << payload_size_bytes
                << " bytes, FF Threads/Process: " << parallel_threads_arg
                << "\n";
      std::cout
          << "Analysis: Dual baseline comparison with MPI scaling isolation\n";
    }
  }

  try {
    double sequential_time = 0.0;
    double parallel_time = 0.0;

    if (rank == 0) {
      if (!skip_baselines) {
        // Establish dual baselines for comprehensive analysis
        sequential_time = run_sequential_baseline(config, csv_ptr);
        if (size == 1) {
          parallel_time =
              run_parallel_baseline(config, sequential_time, csv_ptr);
        }
      } else if (!csv_filename.empty()) {
        // Load existing baselines from CSV
        auto baselines = read_dual_baseline_from_csv(
            csv_filename, static_cast<int>(parallel_threads_arg),
            config.data_size, config.payload_size);
        sequential_time = baselines.first;
        parallel_time = baselines.second;
      }

      // Show table headers only if not in quiet mode
      if (!quiet_mode) {
        std::cout << "\n"
                  << std::left << std::setw(11) << "MPI Procs" << std::right
                  << std::setw(14) << "Time (ms)" << std::right << std::setw(19)
                  << "Throughput (MRec/s)" << std::right << std::setw(12)
                  << "Seq Speedup" << std::right << std::setw(12)
                  << "Par Speedup" << std::right << std::setw(15)
                  << "MPI Eff (%)" << std::right << std::setw(15)
                  << "Total Eff (%)" << std::endl;
        std::cout << std::string(98, '-') << std::endl;
      }
    }

    // Distribute baselines to all processes for consistent calculations
    MPI_Bcast(&sequential_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parallel_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (size > 1 || skip_baselines) {
      run_enhanced_hybrid_benchmark(config, rank, size, sequential_time,
                                    parallel_time, quiet_mode, csv_ptr);
    }

    if (rank == 0 && !quiet_mode) {
      std::cout << "\n" << std::string(98, '-') << std::endl;
      std::cout << "Metrics Explanation:\n";
      std::cout << "• Time (ms): Total execution time in milliseconds\n";
      std::cout << "• Throughput: Million records processed per second\n";
      std::cout << "• Seq Speedup: Performance vs pure sequential std::sort\n";
      std::cout
          << "• Par Speedup: Performance vs single-node parallel (1 MPI + "
          << parallel_threads_arg << " FF threads)\n";
      std::cout << "• MPI Eff (%): How well MPI processes scale (Par Speedup / "
                   "MPI Processes)\n";
      std::cout << "• Total Eff (%): Overall efficiency vs sequential (Seq "
                   "Speedup / Total Threads)\n";
      std::cout << std::string(98, '=') << std::endl;
    }

  } catch (const std::exception &e) {
    if (rank == 0) {
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
