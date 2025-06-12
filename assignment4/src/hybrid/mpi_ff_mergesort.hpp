/**
 * @file mpi_ff_mergesort.hpp
 * @brief Hybrid MPI+FastFlow distributed mergesort with
 * computation-communication overlap
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
 * 3. Hierarchical merging via binary tree reduction with
 * computation-communication overlap
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
   * @brief Hierarchical merge using binary tree reduction with
   * true computation-communication overlap
   * @param local_data Process-local sorted data, becomes final result on rank 0
   */
  void hierarchical_merge_with_overlap(std::vector<Record> &local_data);

  /**
   * @brief Initiate non-blocking receive for maximum overlap opportunity
   */
  void initiate_receive_with_overlap(int source, 
                                     std::vector<MPI_Request> &requests,
                                     std::vector<std::vector<Record>> &buffers,
                                     std::vector<int> &sources);

  /**
   * @brief Process all pending receives with computation overlap
   */
  void process_pending_receives_with_overlap(
      std::vector<Record> &local_data,
      std::vector<MPI_Request> &requests,
      std::vector<std::vector<Record>> &buffers,
      const std::vector<int> &sources);

  /**
   * @brief Receive and merge using true computation-communication overlap
   */
  void receive_and_merge_with_overlap(std::vector<Record> &local_data,
                                      int source);

  /**
   * @brief Send data using true non-blocking MPI for maximum overlap
   */
  void send_data_with_overlap(const std::vector<Record> &data, int target);

  /**
   * @brief Perform useful work while send operation is in progress
   */
  void perform_sender_cleanup_work();

  /**
   * @brief Wait for send completion with continued overlap opportunities
   */
  void wait_for_send_completion_with_overlap(MPI_Request &request);

  /**
   * @brief Process a completed receive operation
   */
  void process_completed_receive(size_t request_index, 
                                 std::vector<Record> &partner_data, int source);

  /**
   * @brief Optimize local data structure during communication wait
   */
  void optimize_local_data_structure(std::vector<Record> &local_data);

  /**
   * @brief Prefetch memory for merge operations
   */
  void prefetch_merge_memory(const std::vector<Record> &local_data);

  /**
   * @brief Cleanup completed MPI requests
   */
  void cleanup_completed_requests(std::vector<MPI_Request> &requests);

  /**
   * @brief Efficient two-way merge with move semantics optimization
   */
  void merge_two_sorted_arrays(std::vector<Record> &local_data,
                               std::vector<Record> &partner_data);

  /**
   * @brief Update performance metrics for execution phase
   */
  void update_metrics(const std::string &phase, double elapsed_time);

  HybridConfig config_;   ///< Algorithm configuration
  int mpi_rank_;          ///< Current process rank
  int mpi_size_;          ///< Total process count
  size_t payload_size_;   ///< Current record payload size
  HybridMetrics metrics_; ///< Performance metrics
};

} // namespace hybrid
