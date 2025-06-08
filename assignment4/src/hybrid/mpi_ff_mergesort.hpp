/**
 * @file mpi_ff_mergesort.hpp
 * @brief Hybrid MPI+FastFlow distributed mergesort implementation
 *
 * Implements scalable distributed sorting combining MPI inter-node
 * communication with FastFlow intra-node parallelization. Uses binary
 * tree reduction for hierarchical merging with optimized buffer strategies.
 *
 * Algorithm Design:
 * - Data distribution via MPI_Scatterv with load balancing
 * - Local sorting using FastFlow parallel mergesort (external dependency)
 * - Hierarchical merge via binary tree reduction pattern
 * - Zero-payload optimization path for key-only records
 *
 * Threading Model: Requires MPI_THREAD_FUNNELED for FastFlow integration
 * Memory Strategy: Contiguous buffer packing for optimal MPI performance
 */

#pragma once

#include "../common/record.hpp"
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace hybrid {

/**
 * @struct HybridConfig
 * @brief Configuration parameters for hybrid mergesort algorithm
 *
 * Controls parallel execution behavior and performance thresholds.
 * Zero parallel_threads triggers automatic detection via hardware_concurrency.
 */
struct HybridConfig {
  size_t parallel_threads{0}; ///< FastFlow worker count (0 = auto-detect)
  size_t min_local_threshold{10000}; ///< Minimum size for parallel local sort
};

/**
 * @struct HybridMetrics
 * @brief Performance metrics collection for algorithm analysis
 *
 * Tracks timing and communication overhead across distributed execution phases.
 * Communication metrics include both MPI overhead and payload transfer costs.
 */
struct HybridMetrics {
  double total_time{0.0};         ///< End-to-end execution time (ms)
  double local_sort_time{0.0};    ///< FastFlow sorting phase time (ms)
  double merge_time{0.0};         ///< Binary tree merge phase time (ms)
  double communication_time{0.0}; ///< Data distribution phase time (ms)
  size_t bytes_communicated{0};   ///< Total bytes transferred via MPI
  size_t local_elements{0};       ///< Final element count (root process only)
};

/**
 * @class HybridMergeSort
 * @brief Distributed mergesort combining MPI distribution with FastFlow
 * parallelization
 *
 * Implements three-phase hybrid sorting algorithm:
 * 1. Data distribution across MPI processes with load balancing
 * 2. Local parallel sorting using FastFlow framework
 * 3. Hierarchical merging via binary tree reduction pattern
 *
 * Performance characteristics:
 * - Time complexity: O(n log n) distributed across P processes
 * - Communication complexity: O(log P) reduction rounds
 * - Memory efficiency: Zero-payload optimization for key-only records
 * - Thread safety: MPI_THREAD_FUNNELED requirement for FastFlow integration
 *
 * Threading model requirements:
 * - MPI initialized with MPI_THREAD_FUNNELED or higher
 * - FastFlow workers execute concurrently on multiple threads
 * - Only main thread performs MPI communication (funneled model)
 */
class HybridMergeSort {
public:
  /**
   * @brief Constructor requiring MPI_THREAD_FUNNELED threading support
   *
   * MPI_THREAD_FUNNELED requirement justification:
   * - FastFlow workers execute on multiple threads simultaneously
   * - Only main thread performs MPI communication (funneled model)
   * - Higher threading levels (MULTIPLE) unnecessary and may hurt performance
   * - Lower levels (SINGLE/SERIALIZED) incompatible with FastFlow execution
   *
   * @param config Algorithm configuration and performance parameters
   * @throws std::runtime_error if MPI threading support insufficient
   */
  explicit HybridMergeSort(const HybridConfig &config);
  ~HybridMergeSort();

  HybridMergeSort(const HybridMergeSort &) = delete;
  HybridMergeSort &operator=(const HybridMergeSort &) = delete;
  HybridMergeSort(HybridMergeSort &&) = default;
  HybridMergeSort &operator=(HybridMergeSort &&) = default;

  /**
   * @brief Distributed sort with hybrid MPI+FastFlow parallelization
   *
   * Three-phase execution:
   * 1. Root broadcasts and scatters data across all processes
   * 2. Each process sorts local partition using FastFlow parallelization
   * 3. Binary tree reduction merges results back to root process
   *
   * @param data Input dataset (root process only, others pass empty vector)
   * @param payload_size Record payload size in bytes for packing optimization
   * @return Sorted dataset on root process, empty on others
   *
   * @throws std::runtime_error if MPI threading support insufficient
   * @pre MPI initialized with MPI_THREAD_FUNNELED or higher
   * @post Root process contains globally sorted data, others empty
   */
  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);

  /**
   * @brief Access performance metrics after sort completion
   * @return Immutable reference to timing and communication metrics
   */
  const HybridMetrics &get_metrics() const { return metrics_; }

private:
  /**
   * @brief Optimized data distribution with payload-aware strategy
   *
   * Performance optimizations:
   * - Zero payload: Uses MPI_UNSIGNED_LONG scatter for cache efficiency
   * - Non-zero payload: Contiguous buffer packing minimizes MPI overhead
   * - Pre-allocation with emplace_back() prevents reallocations
   * - Load balancing: remainder distributed across first (remainder) processes
   */
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);

  /**
   * @brief Local sorting with FastFlow integration and threshold logic
   *
   * Delegates to FastFlow parallel_mergesort for large datasets exceeding
   * min_local_threshold, otherwise uses std::sort for optimal small-data
   * performance.
   */
  void sort_local_data(std::vector<Record> &data);

  /**
   * @brief Hierarchical merge using binary tree reduction pattern
   *
   * Implements log₂(P) communication rounds where P = MPI process count.
   * Each round halves active processes: survivors receive and merge data
   * from partners, then advance to next level. Survivors determined by
   * rank % (2 * step) == 0, ensuring power-of-2 reduction tree.
   *
   * Performance optimizations:
   * - Zero-payload path uses MPI_UNSIGNED_LONG for cache efficiency
   * - Non-zero payload uses contiguous buffer packing for MPI performance
   * - Move semantics during merge operations minimize copy overhead
   *
   * @param local_data Process-local sorted data, becomes final result on rank 0
   * @note Final result available only on rank 0, other processes cleared
   */
  void hierarchical_merge(std::vector<Record> &local_data);

  std::vector<Record> merge_with_partner(std::vector<Record> &local_data,
                                         int partner_rank, bool is_receiver,
                                         MPI_Comm comm);

  /**
   * @brief Update performance metrics for specific execution phase
   * @param phase Phase identifier ("distribution", "local_sort", "merge")
   * @param elapsed_time Phase execution time in milliseconds
   */
  void update_metrics(const std::string &phase, double elapsed_time);

  // Static utility methods for record serialization (currently unused)
  static std::vector<char> pack_records(const std::vector<Record> &records,
                                        size_t record_byte_size);
  static std::vector<Record> unpack_records(const std::vector<char> &buffer,
                                            size_t payload_size,
                                            size_t record_byte_size);
  static std::vector<Record> merge_sorted_vectors(std::vector<Record> &left,
                                                  std::vector<Record> &right);

  HybridConfig config_;   ///< Algorithm configuration parameters
  int mpi_rank_;          ///< Current process rank in MPI_COMM_WORLD
  int mpi_size_;          ///< Total process count in MPI_COMM_WORLD
  size_t payload_size_;   ///< Current record payload size (bytes)
  HybridMetrics metrics_; ///< Performance metrics accumulator
};

} // namespace hybrid
