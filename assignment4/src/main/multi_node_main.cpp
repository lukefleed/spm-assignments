#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../hybrid/mpi_ff_mergesort.hpp"
#include "../sequential/sequential_mergesort.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>

/**
 * @brief Configuration for multi-node execution
 */
struct MultiNodeConfig {
  size_t array_size;
  size_t payload_size;
  size_t ff_threads;
  DataPattern pattern;
  bool validate;
  bool verbose;
  bool benchmark_mode;

  MultiNodeConfig()
      : array_size(1000000), payload_size(64), ff_threads(0),
        pattern(DataPattern::RANDOM), validate(true), verbose(false),
        benchmark_mode(false) {}
};

/**
 * @brief Parse command line arguments for multi-node execution
 */
MultiNodeConfig parse_multi_node_args(int argc, char *argv[]) {
  MultiNodeConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-s" && i + 1 < argc) {
      std::string size_str = argv[++i];
      config.array_size = parse_size(size_str);
    } else if (arg == "-r" && i + 1 < argc) {
      config.payload_size = std::stoul(argv[++i]);
    } else if (arg == "-t" && i + 1 < argc) {
      config.ff_threads = std::stoul(argv[++i]);
    } else if (arg == "-p" && i + 1 < argc) {
      std::string pattern = argv[++i];
      if (pattern == "random")
        config.pattern = DataPattern::RANDOM;
      else if (pattern == "sorted")
        config.pattern = DataPattern::SORTED;
      else if (pattern == "reverse")
        config.pattern = DataPattern::REVERSE_SORTED;
      else if (pattern == "nearly")
        config.pattern = DataPattern::NEARLY_SORTED;
    } else if (arg == "--no-validate") {
      config.validate = false;
    } else if (arg == "--verbose" || arg == "-v") {
      config.verbose = true;
    } else if (arg == "--benchmark" || arg == "-b") {
      config.benchmark_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  -s SIZE     Array size (e.g., 10M, 100M)\n"
          << "  -r BYTES    Record payload size in bytes\n"
          << "  -t THREADS  Number of FastFlow threads per node\n"
          << "  -p PATTERN  Data pattern: random, sorted, reverse, nearly\n"
          << "  --no-validate  Disable result validation\n"
          << "  --verbose   Enable verbose output\n"
          << "  --benchmark Enable benchmark mode\n"
          << "  --help      Show this help message\n";
      MPI_Finalize();
      exit(0);
    }
  }

  return config;
}

/**
 * @brief Validates the hybrid mergesort result
 */
bool validate_hybrid_result(const std::vector<Record> &sorted_data,
                            const std::vector<Record> &original_data,
                            int rank) {
  if (rank != 0)
    return true; // Only validate on root

  if (sorted_data.size() != original_data.size()) {
    std::cerr << "[!] Validation Error: Size mismatch! Expected "
              << original_data.size() << ", got " << sorted_data.size() << "\n";
    return false;
  }

  // Check if sorted
  for (size_t i = 1; i < sorted_data.size(); ++i) {
    if (sorted_data[i] < sorted_data[i - 1]) {
      std::cerr << "[!] Validation Error: Output is not sorted at position "
                << i << "\n";
      return false;
    }
  }

  // Check if it's a permutation (key-based check)
  std::vector<unsigned long> orig_keys, sorted_keys;
  orig_keys.reserve(original_data.size());
  sorted_keys.reserve(sorted_data.size());

  for (const auto &record : original_data) {
    orig_keys.push_back(record.key);
  }
  for (const auto &record : sorted_data) {
    sorted_keys.push_back(record.key);
  }

  std::sort(orig_keys.begin(), orig_keys.end());
  std::sort(sorted_keys.begin(), sorted_keys.end());

  if (orig_keys != sorted_keys) {
    std::cerr << "[!] Validation Error: Key content mismatch after sorting.\n";
    return false;
  }

  return true;
}

/**
 * @brief Print performance summary across all MPI processes
 */
void print_performance_summary(const hybrid::HybridMetrics &metrics,
                               const MultiNodeConfig &config, double total_time,
                               int rank, int size) {
  if (rank == 0) {
    std::cout << "\n=== Multi-Node Hybrid MPI+FastFlow MergeSort Results ===\n";
    std::cout << "Problem Configuration:\n";
    std::cout << "  Array size: " << config.array_size << " elements\n";
    std::cout << "  Payload size: " << config.payload_size << " bytes\n";
    std::cout << "  Total data: "
              << format_bytes(config.array_size *
                              (sizeof(unsigned long) + config.payload_size))
              << "\n";
    std::cout << "  MPI processes: " << size << "\n";
    std::cout << "  FastFlow threads per node: " << config.ff_threads << "\n";
    std::cout << "  Data pattern: ";
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

    std::cout << "\nPerformance Results:\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total execution time: " << total_time << " ms\n";
    std::cout << "  Local sort time: " << metrics.local_sort_time << " ms\n";
    std::cout << "  Communication time: " << metrics.communication_time
              << " ms\n";
    std::cout << "  Merge time: " << metrics.merge_time << " ms\n";
    std::cout << "  Load balance ratio: " << std::setprecision(3)
              << metrics.load_balance_ratio << "\n";
    std::cout << "  Data communicated: "
              << format_bytes(metrics.bytes_communicated) << "\n";

    // Calculate efficiency metrics
    double comm_ratio = metrics.communication_time / total_time;
    double compute_ratio = metrics.local_sort_time / total_time;

    std::cout << "  Communication ratio: " << std::setprecision(1)
              << (comm_ratio * 100) << "%\n";
    std::cout << "  Computation ratio: " << std::setprecision(1)
              << (compute_ratio * 100) << "%\n";

    // Estimate sequential baseline for speedup calculation
    double elements_per_sec = config.array_size / (total_time / 1000.0);
    std::cout << "  Throughput: " << std::setprecision(2)
              << (elements_per_sec / 1e6) << " M elements/sec\n";
  }
}

/**
 * @brief Run benchmark suite for performance analysis
 */
void run_benchmark_suite(const MultiNodeConfig &base_config, int rank,
                         int size) {
  if (rank == 0) {
    std::cout << "\n=== Benchmark Suite ===\n";
    std::cout << "Running comprehensive performance tests...\n";
  }

  std::vector<size_t> test_sizes = {1000000, 10000000, 100000000};
  std::vector<size_t> payload_sizes = {8, 64, 256};
  std::vector<size_t> thread_counts = {1, 4, 8, 16};

  for (size_t test_size : test_sizes) {
    for (size_t payload_size : payload_sizes) {
      for (size_t threads : thread_counts) {
        if (threads > hybrid::utils::get_optimal_ff_threads())
          continue;

        MultiNodeConfig test_config = base_config;
        test_config.array_size = test_size;
        test_config.payload_size = payload_size;
        test_config.ff_threads = threads;
        test_config.validate = false; // Skip validation for speed

        // Generate test data on root
        std::vector<Record> test_data;
        if (rank == 0) {
          test_data =
              generate_records(test_config.array_size, test_config.payload_size,
                               test_config.pattern);
        }

        Timer timer;
        hybrid::HybridConfig hybrid_config;
        hybrid_config.ff_threads = test_config.ff_threads;

        hybrid::HybridMergeSort sorter(hybrid_config);
        auto result = sorter.sort(test_data, test_config.payload_size);
        double elapsed = timer.elapsed_ms();

        const auto &metrics = sorter.get_metrics();

        if (rank == 0) {
          std::cout << std::fixed << std::setprecision(2);
          std::cout << "Size: " << (test_size / 1e6) << "M, "
                    << "Payload: " << payload_size << "B, "
                    << "Threads: " << threads << ", "
                    << "Time: " << elapsed << "ms, "
                    << "Throughput: " << (test_size / (elapsed / 1000.0) / 1e6)
                    << "M/s\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  // Initialize MPI with thread support
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "Error: MPI implementation does not provide required thread "
                 "support\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    MultiNodeConfig config = parse_multi_node_args(argc, argv);

    if (rank == 0 && config.verbose) {
      std::cout << "Hybrid MPI+FastFlow MergeSort starting...\n";
      std::cout << "MPI processes: " << size << "\n";
      std::cout << "FastFlow threads per node: "
                << (config.ff_threads == 0
                        ? hybrid::utils::get_optimal_ff_threads()
                        : config.ff_threads)
                << "\n";
    }

    // Run benchmark suite if requested
    if (config.benchmark_mode) {
      run_benchmark_suite(config, rank, size);
      MPI_Finalize();
      return 0;
    }

    // Generate test data (only on root process)
    std::vector<Record> original_data;
    if (rank == 0) {
      if (config.verbose) {
        std::cout << "Generating " << config.array_size << " records with "
                  << config.payload_size << " byte payload...\n";
      }

      original_data = generate_records(config.array_size, config.payload_size,
                                       config.pattern);

      if (config.verbose) {
        std::cout << "Data generation complete. Starting hybrid sort...\n";
      }
    }

    // Create hybrid sorter configuration
    hybrid::HybridConfig hybrid_config;
    hybrid_config.ff_threads = config.ff_threads;
    hybrid_config.enable_overlap = true;
    hybrid_config.use_async_merging = true;
    hybrid_config.load_balance_factor = 0.1;

    // Adjust configuration based on problem size and cluster size
    size_t total_threads =
        size * (config.ff_threads == 0 ? hybrid::utils::get_optimal_ff_threads()
                                       : config.ff_threads);

    if (config.array_size < total_threads * 1024) {
      hybrid_config.enable_overlap = false;
      hybrid_config.use_async_merging = false;
      hybrid_config.load_balance_factor = 0.0;
    }

    // Execute hybrid mergesort
    Timer total_timer;
    hybrid::HybridMergeSort sorter(hybrid_config);
    auto sorted_data = sorter.sort(original_data, config.payload_size);
    double total_time = total_timer.elapsed_ms();

    const auto &metrics = sorter.get_metrics();

    // Validate result if requested
    bool is_valid = true;
    if (config.validate) {
      is_valid = validate_hybrid_result(sorted_data, original_data, rank);
      if (rank == 0) {
        std::cout << "Validation: " << (is_valid ? "✓ PASSED" : "✗ FAILED")
                  << "\n";
      }
    }

    // Print performance summary
    if (!config.benchmark_mode) {
      print_performance_summary(metrics, config, total_time, rank, size);
    }

    // Compare with sequential baseline (optional, on small datasets)
    if (rank == 0 && config.array_size <= 10000000 && config.verbose) {
      std::cout << "\nRunning sequential comparison...\n";
      auto seq_data = copy_records(original_data);
      Timer seq_timer;
      sequential_mergesort(seq_data);
      double seq_time = seq_timer.elapsed_ms();

      double speedup = seq_time / total_time;
      std::cout << "Sequential time: " << std::fixed << std::setprecision(2)
                << seq_time << " ms\n";
      std::cout << "Hybrid speedup: " << std::setprecision(2) << speedup
                << "x\n";

      if (speedup >= 1.0) {
        std::cout << "✓ Positive speedup achieved!\n";
      } else {
        std::cout << "⚠ Negative speedup - consider tuning parameters\n";
      }
    }

  } catch (const std::exception &e) {
    if (rank == 0) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
