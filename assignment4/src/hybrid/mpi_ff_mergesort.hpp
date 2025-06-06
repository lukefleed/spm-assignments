#ifndef MPI_FF_MERGESORT_HPP
#define MPI_FF_MERGESORT_HPP

#include "../common/record.hpp"
#include "mpi_communication.hpp"
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @brief High-performance hybrid MPI+FastFlow mergesort implementation
 *
 * Combines the power of distributed computing via MPI with intra-node
 * parallelization using FastFlow for maximum performance scalability.
 */
namespace hybrid {

/**
 * @brief Configuration for hybrid mergesort execution
 */
struct HybridConfig {
  size_t ff_threads; ///< Number of FastFlow threads per node
  size_t
      min_local_threshold; ///< Minimum elements for local FastFlow processing
  bool enable_overlap;     ///< Enable computation-communication overlap
  bool use_async_merging;  ///< Use asynchronous merging in tree phases
  double load_balance_factor; ///< Load balancing sensitivity (0.0-1.0)

  HybridConfig()
      : ff_threads(0), min_local_threshold(4096), enable_overlap(true),
        use_async_merging(true), load_balance_factor(0.1) {}
};

/**
 * @brief Performance metrics for hybrid mergesort execution
 */
struct HybridMetrics {
  double total_time;         ///< Total execution time
  double local_sort_time;    ///< Time spent in local FastFlow sorting
  double communication_time; ///< Time spent in MPI communication
  double merge_time;         ///< Time spent in distributed merging
  double load_balance_ratio; ///< Load balancing efficiency ratio
  size_t bytes_communicated; ///< Total bytes transferred via MPI
  size_t local_elements;     ///< Elements processed locally

  HybridMetrics()
      : total_time(0), local_sort_time(0), communication_time(0), merge_time(0),
        load_balance_ratio(1.0), bytes_communicated(0), local_elements(0) {}
};

/**
 * @brief Main hybrid MPI+FastFlow mergesort class
 *
 * Implements a highly optimized distributed mergesort algorithm that:
 * 1. Distributes data optimally across MPI processes
 * 2. Uses FastFlow for high-performance local sorting
 * 3. Implements hierarchical merging with computation-communication overlap
 * 4. Provides extensive performance monitoring and optimization
 */
class HybridMergeSort {
private:
  int mpi_rank_;
  int mpi_size_;
  HybridConfig config_;
  std::unique_ptr<mpi_comm::RecordDatatype> datatype_;
  std::unique_ptr<mpi_comm::AsyncCommManager> comm_manager_;
  HybridMetrics metrics_;

  /**
   * @brief Performs initial data distribution across MPI processes
   */
  void distribute_data(std::vector<Record> &local_data,
                       const std::vector<Record> &global_data);

  /**
   * @brief Executes local FastFlow-based sorting
   */
  void sort_local_data(std::vector<Record> &local_data);

  /**
   * @brief Implements hierarchical distributed merging
   */
  void hierarchical_merge(std::vector<Record> &local_data);

  /**
   * @brief Performs load balancing across processes
   */
  void balance_load(std::vector<Record> &local_data);

  /**
   * @brief Optimized pairwise merge operation with overlap
   */
  std::vector<Record> merge_with_partner(const std::vector<Record> &local_data,
                                         int partner_rank, bool is_receiver);

  /**
   * @brief Gathers final sorted data to root process
   */
  void gather_final_result(std::vector<Record> &result_data,
                           const std::vector<Record> &local_data);

  /**
   * @brief Updates performance metrics during execution
   */
  void update_metrics(const std::string &phase, double elapsed_time,
                      size_t bytes_transferred = 0);

  /**
   * @brief Helper function to merge two sorted vectors of Records
   */
  std::vector<Record> merge_sorted_vectors(const std::vector<Record> &left,
                                           std::vector<Record> &right);

public:
  /**
   * @brief Constructor with automatic MPI initialization detection
   */
  explicit HybridMergeSort(const HybridConfig &config = HybridConfig());

  /**
   * @brief Destructor with cleanup
   */
  ~HybridMergeSort();

  /**
   * @brief Main entry point for hybrid mergesort
   *
   * @param data Input data vector (only meaningful on root process)
   * @param payload_size Size of record payload in bytes
   * @return Sorted data (only on root process), empty elsewhere
   */
  std::vector<Record> sort(std::vector<Record> &data, size_t payload_size);

  /**
   * @brief Returns performance metrics from last sort operation
   */
  const HybridMetrics &get_metrics() const { return metrics_; }

  /**
   * @brief Updates configuration for next sort operation
   */
  void set_config(const HybridConfig &config) { config_ = config; }

  /**
   * @brief Returns current MPI rank
   */
  int rank() const { return mpi_rank_; }

  /**
   * @brief Returns total MPI processes
   */
  int size() const { return mpi_size_; }

  /**
   * @brief Validates algorithm correctness (debug builds only)
   */
  bool validate_result(const std::vector<Record> &sorted_data,
                       const std::vector<Record> &original_data) const;
};

/**
 * @brief Utility functions for hybrid mergesort
 */
namespace utils {

/**
 * @brief Determines optimal FastFlow thread count based on system
 */
size_t get_optimal_ff_threads();

/**
 * @brief Estimates communication cost for given data size
 */
double estimate_comm_cost(size_t data_size, size_t payload_size,
                          int num_processes);

/**
 * @brief Calculates optimal data distribution strategy
 */
std::vector<size_t> calculate_optimal_distribution(size_t total_elements,
                                                   int num_processes,
                                                   double load_factor = 0.1);

/**
 * @brief Validates MPI environment for hybrid execution
 */
bool validate_mpi_environment();

/**
 * @brief Benchmarks local FastFlow performance
 */
double benchmark_fastflow_performance(size_t test_size, size_t payload_size,
                                      size_t num_threads);
} // namespace utils

/**
 * @brief High-level convenience function for hybrid mergesort
 *
 * Provides a simple interface for users who don't need fine-grained control.
 * Automatically determines optimal configuration based on problem size and
 * available resources.
 *
 * @param data Input data (meaningful only on MPI rank 0)
 * @param payload_size Record payload size in bytes
 * @param ff_threads Number of FastFlow threads (0 = auto-detect)
 * @return Sorted data on rank 0, empty elsewhere
 */
std::vector<Record> hybrid_mergesort(std::vector<Record> &data,
                                     size_t payload_size,
                                     size_t ff_threads = 0);

} // namespace hybrid

#endif // MPI_FF_MERGESORT_HPP
