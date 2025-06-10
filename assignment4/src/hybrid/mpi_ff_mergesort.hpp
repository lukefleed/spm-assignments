/**
 * @file mpi_ff_mergesort.hpp
 * @brief Hybrid MPI+FastFlow distributed mergesort
 */

#pragma once

#include "../common/record.hpp"
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace hybrid {

/**
 * @brief Configuration parameters for hybrid mergesort
 */
struct HybridConfig {
  size_t parallel_threads{0}; ///< FastFlow worker count (0 = auto-detect)
  size_t min_local_threshold{10000}; ///< Minimum size for parallel local sort
};

/**
 * @brief Performance metrics collection
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
 * @brief Distributed mergesort combining MPI distribution with FastFlow
 * parallelization
 *
 * Three-phase algorithm:
 * 1. Data distribution across MPI processes
 * 2. Local parallel sorting using FastFlow
 * 3. Hierarchical merging via binary tree reduction
 *
 * Requires MPI_THREAD_FUNNELED for FastFlow integration.
 */
class HybridMergeSort {
public:
  /**
   * @brief Initialize hybrid sorter
   * @param config Algorithm configuration parameters
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
   * @param data Input dataset (root process only, others pass empty vector)
   * @param payload_size Record payload size in bytes
   * @return Sorted dataset on root process, empty on others
   */
  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);

  /**
   * @brief Get performance metrics after sort completion
   */
  const HybridMetrics &get_metrics() const { return metrics_; }

private:
  /**
   * @brief Distribute data across MPI processes with load balancing
   */
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);

  /**
   * @brief Sort local data using FastFlow or std::sort based on size threshold
   */
  void sort_local_data(std::vector<Record> &data);

  /**
   * @brief Hierarchical merge using binary tree reduction pattern
   * @param local_data Process-local sorted data, becomes final result on rank 0
   */
  void hierarchical_merge(std::vector<Record> &local_data);

  std::vector<Record> merge_with_partner(std::vector<Record> &local_data,
                                         int partner_rank, bool is_receiver,
                                         MPI_Comm comm);

  /**
   * @brief Update performance metrics for execution phase
   */
  void update_metrics(const std::string &phase, double elapsed_time);

  // Utility methods for record serialization
  static std::vector<char> pack_records(const std::vector<Record> &records,
                                        size_t record_byte_size);
  static std::vector<Record> unpack_records(const std::vector<char> &buffer,
                                            size_t payload_size,
                                            size_t record_byte_size);
  static std::vector<Record> merge_sorted_vectors(std::vector<Record> &left,
                                                  std::vector<Record> &right);

  HybridConfig config_;   ///< Algorithm configuration
  int mpi_rank_;          ///< Current process rank
  int mpi_size_;          ///< Total process count
  size_t payload_size_;   ///< Current record payload size
  HybridMetrics metrics_; ///< Performance metrics
};

} // namespace hybrid
