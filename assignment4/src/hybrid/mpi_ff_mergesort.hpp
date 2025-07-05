#pragma once

#include "../common/record.hpp"
#include <mpi.h>
#include <string>
#include <vector>

namespace hybrid {

/**
 * @brief Configuration parameters for hybrid mergesort.
 */
struct HybridConfig {
  size_t parallel_threads{0}; ///< Number of threads per process.
  size_t min_local_threshold{
      10000}; ///< Minimum data size for parallel local sort.
};

/**
 * @brief Performance metrics collection.
 */
struct HybridMetrics {
  double total_time{0.0};         ///< End-to-end execution time (ms).
  double local_sort_time{0.0};    ///< Local sorting phase time (ms).
  double merge_time{0.0};         ///< Hierarchical merge phase time (ms).
  double communication_time{0.0}; ///< Data distribution phase time (ms).
  size_t bytes_communicated{0};   ///< Total bytes transferred via MPI.
  size_t local_elements{0};       ///< Final element count (root process only).
};

/**
 * @brief Distributed mergesort combining MPI with thread-based parallelism.
 *
 * Implements a three-phase algorithm:
 * 1. Data distribution across MPI processes using a load-balanced scatter.
 * 2. Local parallel sorting on each node.
 * 3. A pipelined hierarchical merge using a binary-tree reduction pattern.
 *    This phase employs non-blocking receives to overlap computation
 *    (merging) and communication (data transfer).
 */
class HybridMergeSort {
public:
  /**
   * @brief Initializes the hybrid sorter, verifying the MPI environment.
   * @param config Configuration parameters for the sort.
   */
  explicit HybridMergeSort(const HybridConfig &config);
  ~HybridMergeSort();

  HybridMergeSort(const HybridMergeSort &) = delete;
  HybridMergeSort &operator=(const HybridMergeSort &) = delete;
  HybridMergeSort(HybridMergeSort &&) = default;
  HybridMergeSort &operator=(HybridMergeSort &&) = default;

  /**
   * @brief Executes the distributed hybrid sort.
   * @param data The global vector of records (only used by root process).
   * @param payload_size The size of the payload for each record.
   * @return A sorted vector of records on the root process (rank 0).
   */
  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);

  /**
   * @brief Retrieves performance metrics after a sort operation.
   * @return A const reference to the collected metrics.
   */
  const HybridMetrics &get_metrics() const { return metrics_; }

private:
  // Main phases of the distributed sort
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);
  void sort_local_data(std::vector<Record> &data);
  void hierarchical_merge(std::vector<Record> &local_data);

  // A helper for high-performance intra-node parallel merging.
  void parallel_merge(std::vector<Record> &local_data,
                      std::vector<Record> &partner_data);

  // Utility for metrics collection.
  void update_metrics(const std::string &phase, double elapsed_time);

  HybridConfig config_;
  int mpi_rank_;
  int mpi_size_;
  size_t payload_size_;
  HybridMetrics metrics_;
};

} // namespace hybrid
