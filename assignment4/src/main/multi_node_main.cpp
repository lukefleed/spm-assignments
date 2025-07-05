#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "../hybrid/mpi_ff_mergesort.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <vector>

/**
 * @brief Multi-node execution configuration
 */
struct MultiNodeConfig {
  size_t array_size;
  size_t payload_size;
  size_t parallel_threads;
  DataPattern pattern;

  MultiNodeConfig()
      : array_size(1000000), payload_size(64), parallel_threads(4),
        pattern(DataPattern::RANDOM) {}
};

/**
 * @brief Parse command line arguments for multi-node execution
 */
MultiNodeConfig parse_multi_node_args(int argc, char *argv[]) {
  MultiNodeConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-s" && i + 1 < argc) {
      config.array_size = parse_size(argv[++i]);
    } else if (arg == "-r" && i + 1 < argc) {
      config.payload_size = std::stoul(argv[++i]);
    } else if (arg == "-t" && i + 1 < argc) {
      config.parallel_threads = std::stoul(argv[++i]);
    } else if (arg == "-p" && i + 1 < argc) {
      std::string pattern_str = argv[++i];
      if (pattern_str == "sorted")
        config.pattern = DataPattern::SORTED;
      else if (pattern_str == "reverse")
        config.pattern = DataPattern::REVERSE_SORTED;
      else if (pattern_str == "nearly")
        config.pattern = DataPattern::NEARLY_SORTED;
      else
        config.pattern = DataPattern::RANDOM;
    }
  }
  return config;
}

/**
 * @brief Validate hybrid mergesort result correctness.
 *
 * This version validates the sorted data against a pre-sorted list of original
 * keys, which avoids the need to copy the full Record vector.
 *
 * @param sorted_data The final sorted vector of records.
 * @param original_keys_sorted A vector containing all keys from the original
 * data, sorted in ascending order.
 * @param rank The MPI rank of the calling process.
 * @return True if validation passes, false otherwise.
 */
bool validate_hybrid_result(
    const std::vector<Record> &sorted_data,
    const std::vector<unsigned long> &original_keys_sorted, int rank) {
  if (rank != 0)
    return true;

  if (sorted_data.size() != original_keys_sorted.size()) {
    std::cerr << "[!] Validation Error: Size mismatch! Expected "
              << original_keys_sorted.size() << ", got " << sorted_data.size()
              << "\n";
    return false;
  }

  // Check key content and order simultaneously.
  for (size_t i = 0; i < sorted_data.size(); ++i) {
    if (sorted_data[i].key != original_keys_sorted[i]) {
      std::cerr << "[!] Validation Error: Key mismatch or incorrect order at "
                   "position "
                << i << ". Expected key " << original_keys_sorted[i] << ", got "
                << sorted_data[i].key << ".\n";
      return false;
    }
  }

  return true;
}

/**
 * @brief Print a simple performance summary.
 */
void print_performance_summary(const hybrid::HybridMetrics &metrics,
                               const MultiNodeConfig &config, double total_time,
                               int rank, int size) {
  if (rank == 0) {
    std::cout << "\n=== Multi-Node Hybrid MPI+Parallel Sort Results ===\n";
    std::cout << "  Array size: " << config.array_size << " elements\n";
    std::cout << "  Payload size: " << config.payload_size << " bytes\n";
    std::cout << "  MPI processes: " << size << "\n";
    std::cout << "  Threads per node: " << config.parallel_threads << "\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total execution time: " << total_time << " ms\n";
    std::cout << "==================================================\n";
  }
}

/**
 * @brief Multi-node hybrid mergesort main application.
 */
int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr
        << "MPI implementation does not provide required thread support\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    MultiNodeConfig config = parse_multi_node_args(argc, argv);
    std::vector<Record> data_to_sort;
    std::vector<unsigned long> original_keys;

    // Rank 0 generates data and extracts keys for validation.
    if (rank == 0) {
      data_to_sort =
          generate_data(config.array_size, config.payload_size, config.pattern);
      original_keys.reserve(data_to_sort.size());
      for (const auto &rec : data_to_sort) {
        original_keys.push_back(rec.key);
      }
      // Sort the keys to create a ground truth for validation.
      std::sort(original_keys.begin(), original_keys.end());
    }

    hybrid::HybridConfig hybrid_config;
    hybrid_config.parallel_threads = config.parallel_threads;
    hybrid::HybridMergeSort sorter(hybrid_config);

    Timer total_timer;
    // Pass the vector to the sorter. Since Record is move-only, the sorter
    // will operate on this data without any illegal copy operations.
    auto sorted_data = sorter.sort(data_to_sort, config.payload_size);
    double total_time = total_timer.elapsed_ms();

    // Validate the result on rank 0.
    if (!validate_hybrid_result(sorted_data, original_keys, rank)) {
      if (rank == 0) {
        std::cerr << "[!] Correctness validation FAILED.\n";
      }
    } else {
      if (rank == 0) {
        std::cout << "[+] Correctness validation PASSED.\n";
      }
    }

    print_performance_summary(sorter.get_metrics(), config, total_time, rank,
                              size);

  } catch (const std::exception &e) {
    if (rank == 0) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
