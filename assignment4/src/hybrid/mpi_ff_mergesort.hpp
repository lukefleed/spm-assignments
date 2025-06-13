/**
 * @file mpi_ff_mergesort.hpp
 * @brief Hybrid MPI+FastFlow distributed mergesort with
 * computation-communication overlap.
 */

#pragma once

#include "../common/record.hpp"
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace hybrid {

/**
 * @brief Configuration parameters for hybrid mergesort.
 */
struct HybridConfig {
  size_t parallel_threads{0}; ///< FastFlow worker count per process.
  size_t min_local_threshold{
      10000}; ///< Minimum data size for parallel local sort.
};

/**
 * @brief Performance metrics collection.
 */
struct HybridMetrics {
  double total_time{0.0};         ///< End-to-end execution time (ms).
  double local_sort_time{0.0};    ///< FastFlow sorting phase time (ms).
  double merge_time{0.0};         ///< Binary tree merge phase time (ms).
  double communication_time{0.0}; ///< Data distribution phase time (ms).
  size_t bytes_communicated{0};   ///< Total bytes transferred via MPI.
  size_t local_elements{0};       ///< Final element count (root process only).
};

/**
 * @brief Distributed mergesort combining MPI distribution with FastFlow
 * parallelization.
 *
 * Implements a three-phase algorithm:
 * 1. Data distribution across MPI processes using a load-balanced scatter.
 * 2. Local parallel sorting on each node using FastFlow.
 * 3. Hierarchical merging via a binary tree reduction pattern, employing
 *    non-blocking communication to overlap computation and data transfers.
 *
 * This class assumes MPI has been initialized with MPI_THREAD_FUNNELED support.
 */
class HybridMergeSort {
public:
  /**
   * @brief Initializes the hybrid sorter, verifying the MPI environment.
   * @param config Algorithm configuration parameters.
   * @throws std::runtime_error if MPI is not initialized or lacks thread
   * support.
   */
  explicit HybridMergeSort(const HybridConfig &config);
  ~HybridMergeSort();

  HybridMergeSort(const HybridMergeSort &) = delete;
  HybridMergeSort &operator=(const HybridMergeSort &) = delete;
  HybridMergeSort(HybridMergeSort &&) = default;
  HybridMergeSort &operator=(HybridMergeSort &&) = default;

  /**
   * @brief Executes the distributed hybrid sort.
   * @param data Input dataset (meaningful only on the root process).
   * @param payload_size Record payload size in bytes.
   * @return Sorted dataset on the root process, empty on others.
   */
  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);

  /**
   * @brief Retrieves performance metrics after a sort operation.
   * @return A const reference to the collected metrics.
   */
  const HybridMetrics &get_metrics() const { return metrics_; }

private:
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);
  void sort_local_data(std::vector<Record> &data);
  void hierarchical_merge(std::vector<Record> &local_data);

  void send_data_and_exit(const std::vector<Record> &data, int target);
  void merge_two_sorted_arrays(std::vector<Record> &local_data,
                               std::vector<Record> &partner_data);

  void update_metrics(const std::string &phase, double elapsed_time);

  HybridConfig config_;
  int mpi_rank_;
  int mpi_size_;
  size_t payload_size_;
  HybridMetrics metrics_;
};

} // namespace hybrid
