/**
 * @file mpi_ff_mergesort.cpp
 * @brief Hybrid MPI+FastFlow distributed mergesort with k-way merge
 * optimization
 */

#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

/**
 * @brief External FastFlow parallel mergesort implementation
 */
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace hybrid {

HybridMergeSort::HybridMergeSort(const HybridConfig &config)
    : config_(config), mpi_rank_(-1), mpi_size_(-1), payload_size_(0),
      metrics_{} {
  // Verify MPI threading support for FastFlow integration
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error("MPI does not support MPI_THREAD_FUNNELED.");
  }

  // Initialize MPI rank and size for this process
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  if (config_.parallel_threads == 0) {
    throw std::invalid_argument(
        "parallel_threads must be explicitly set (> 0)");
  }
}

HybridMergeSort::~HybridMergeSort() = default;

/**
 * @brief Distributed sort with three-phase hybrid algorithm
 */
std::vector<Record> HybridMergeSort::sort(std::vector<Record> &data,
                                          size_t payload_size) {
  Timer total_timer;
  payload_size_ = payload_size;
  std::vector<Record> local_data;

  // Phase 1: Distribute data across all processes
  Timer dist_timer;
  distribute_data(local_data, data);
  update_metrics("distribution", dist_timer.elapsed_ms());

  // Phase 2: Sort local data partition using FastFlow or std::sort
  Timer sort_timer;
  sort_local_data(local_data);
  update_metrics("local_sort", sort_timer.elapsed_ms());

  // Phase 3: Hierarchical merge with minimal overlap
  Timer merge_timer;
  hierarchical_merge_with_overlap(local_data);
  update_metrics("merge", merge_timer.elapsed_ms());

  metrics_.total_time = total_timer.elapsed_ms();
  metrics_.local_elements = (mpi_rank_ == 0) ? local_data.size() : 0;

  return local_data;
}

/**
 * @brief Distribute data across MPI processes with load balancing (UNCHANGED)
 */
void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  // All processes need to know total size for consistent allocation
  size_t total_num_records = (mpi_rank_ == 0) ? global_data.size() : 0;
  MPI_Bcast(&total_num_records, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (total_num_records == 0)
    return;

  // Calculate how many records each process gets (load balanced)
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);

  size_t base_count = total_num_records / mpi_size_; // Base records per process
  size_t remainder =
      total_num_records % mpi_size_; // Extra records to distribute

  for (int i = 0; i < mpi_size_; ++i) {
    // First 'remainder' processes get one extra record
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    // Calculate starting position for each process
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // Pre-allocate local data storage
  local_data.clear();
  local_data.reserve(send_counts[mpi_rank_]);
  for (int i = 0; i < send_counts[mpi_rank_]; ++i) {
    local_data.emplace_back(payload_size_);
  }

  if (payload_size_ == 0) {
    // Zero-payload optimization: only send keys as unsigned long array
    std::vector<unsigned long> keys;
    if (mpi_rank_ == 0) {
      // Root extracts all keys into contiguous array
      keys.reserve(global_data.size());
      for (const auto &rec : global_data) {
        keys.push_back(rec.key);
      }
    }

    // Scatter keys to all processes
    std::vector<unsigned long> local_keys(send_counts[mpi_rank_]);
    MPI_Scatterv(keys.data(), send_counts.data(), displs.data(),
                 MPI_UNSIGNED_LONG, local_keys.data(), send_counts[mpi_rank_],
                 MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Copy received keys into local Records
    for (size_t i = 0; i < local_keys.size(); ++i) {
      local_data[i].key = local_keys[i];
    }
  } else {
    // Non-zero payload: pack complete records into byte buffer
    const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
    std::vector<int> send_counts_bytes(mpi_size_);
    std::vector<int> displs_bytes(mpi_size_);

    // Convert record counts to byte counts for MPI
    for (int i = 0; i < mpi_size_; ++i) {
      send_counts_bytes[i] = send_counts[i] * record_byte_size;
      displs_bytes[i] = displs[i] * record_byte_size;
    }

    // Root packs all records into contiguous byte buffer
    std::vector<char> send_buffer;
    if (mpi_rank_ == 0) {
      send_buffer.resize(total_num_records * record_byte_size);
      char *ptr = send_buffer.data();
      for (const auto &rec : global_data) {
        // Pack key first
        memcpy(ptr, &rec.key, sizeof(unsigned long));
        ptr += sizeof(unsigned long);
        // Pack payload if present
        if (rec.payload && rec.payload_size > 0) {
          memcpy(ptr, rec.payload, rec.payload_size);
        }
        ptr += rec.payload_size;
      }
    }

    // Scatter packed data to all processes
    std::vector<char> recv_buffer(send_counts_bytes[mpi_rank_]);
    MPI_Scatterv(send_buffer.data(), send_counts_bytes.data(),
                 displs_bytes.data(), MPI_BYTE, recv_buffer.data(),
                 recv_buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Unpack received data into local Records
    const char *ptr = recv_buffer.data();
    for (int i = 0; i < send_counts[mpi_rank_]; ++i) {
      // Unpack key first
      memcpy(&local_data[i].key, ptr, sizeof(unsigned long));
      ptr += sizeof(unsigned long);
      // Unpack payload if present
      if (payload_size_ > 0) {
        memcpy(local_data[i].payload, ptr, payload_size_);
      }
      ptr += payload_size_;
    }
  }

  metrics_.bytes_communicated +=
      send_counts[mpi_rank_] * (sizeof(unsigned long) + payload_size_);
}

/**
 * @brief Sort local data using FastFlow or std::sort (UNCHANGED)
 */
void HybridMergeSort::sort_local_data(std::vector<Record> &data) {
  if (data.empty())
    return;

  // Use FastFlow for large datasets, std::sort for small ones to avoid overhead
  if (data.size() >= config_.min_local_threshold &&
      config_.parallel_threads > 1) {
    parallel_mergesort(data, config_.parallel_threads);
  } else {
    std::sort(data.begin(), data.end());
  }
}

/**
 * @brief OPTIMIZED: K-way merge with single communication round + FastFlow
 * parallelization
 */
void HybridMergeSort::hierarchical_merge_with_overlap(
    std::vector<Record> &local_data) {
  // REVOLUTIONARY CHANGE: Replace binary tree with k-way merge
  // This reduces communication rounds from log(P) to 1

  if (mpi_rank_ == 0) {
    // ROOT PROCESS: Collect all partitions and perform k-way merge
    k_way_merge_as_root(local_data);
  } else {
    // NON-ROOT PROCESSES: Send data to root with overlap
    send_partition_to_root(local_data);
    local_data.clear(); // Free memory immediately
  }
}

/**
 * @brief Root process: Collect all partitions and perform parallel k-way merge
 */
void HybridMergeSort::k_way_merge_as_root(std::vector<Record> &local_data) {
  // Step 1: Prepare storage for all partitions
  std::vector<std::vector<Record>> all_partitions(mpi_size_);
  all_partitions[0] = std::move(local_data); // Keep root's own data

  // Step 2: Collect sizes from all processes first (for pre-allocation)
  std::vector<size_t> partition_sizes(mpi_size_);
  partition_sizes[0] = all_partitions[0].size();

  for (int source = 1; source < mpi_size_; ++source) {
    MPI_Recv(&partition_sizes[source], 1, MPI_UNSIGNED_LONG, source, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Step 3: Non-blocking receives with overlap
  std::vector<MPI_Request> recv_requests;
  std::vector<std::vector<char>> recv_buffers;
  std::vector<std::vector<unsigned long>> key_buffers; // For zero-payload case

  // Initiate all receives simultaneously for maximum overlap
  for (int source = 1; source < mpi_size_; ++source) {
    size_t size = partition_sizes[source];
    if (size > 0) {
      all_partitions[source].reserve(size);
      for (size_t i = 0; i < size; ++i) {
        all_partitions[source].emplace_back(payload_size_);
      }

      MPI_Request req;
      if (payload_size_ == 0) {
        // Zero-payload: receive keys directly
        key_buffers.emplace_back(size);
        MPI_Irecv(key_buffers.back().data(), size, MPI_UNSIGNED_LONG, source, 1,
                  MPI_COMM_WORLD, &req);
      } else {
        // Non-zero payload: receive packed data
        const size_t record_bytes = sizeof(unsigned long) + payload_size_;
        recv_buffers.emplace_back(size * record_bytes);
        MPI_Irecv(recv_buffers.back().data(), recv_buffers.back().size(),
                  MPI_BYTE, source, 1, MPI_COMM_WORLD, &req);
      }
      recv_requests.push_back(req);
    }
  }

  // Step 4: OVERLAP - While waiting for receives, prepare k-way merge
  // structures
  size_t total_elements = 0;
  for (size_t size : partition_sizes) {
    total_elements += size;
  }

  // Pre-allocate final result
  std::vector<Record> final_result;
  final_result.reserve(total_elements);

  // Step 5: Wait for all receives and unpack data
  for (size_t i = 0; i < recv_requests.size(); ++i) {
    MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);

    int source = i + 1; // recv_requests[i] corresponds to source i+1
    if (partition_sizes[source] > 0) {
      if (payload_size_ == 0) {
        // Unpack keys
        const auto &keys = key_buffers[i];
        for (size_t j = 0; j < partition_sizes[source]; ++j) {
          all_partitions[source][j].key = keys[j];
        }
      } else {
        // Unpack complete records
        const char *ptr = recv_buffers[i].data();
        for (size_t j = 0; j < partition_sizes[source]; ++j) {
          memcpy(&all_partitions[source][j].key, ptr, sizeof(unsigned long));
          ptr += sizeof(unsigned long);
          if (payload_size_ > 0) {
            memcpy(all_partitions[source][j].payload, ptr, payload_size_);
          }
          ptr += payload_size_;
        }
      }
    }
  }

  // Step 6: FastFlow-parallelized k-way merge
  local_data = parallel_k_way_merge(all_partitions);

  // Update metrics
  metrics_.bytes_communicated += (total_elements - partition_sizes[0]) *
                                 (sizeof(unsigned long) + payload_size_);
}

/**
 * @brief Non-root processes: Send partition to root with overlap
 */
void HybridMergeSort::send_partition_to_root(const std::vector<Record> &data) {
  // Send size first
  size_t size = data.size();
  MPI_Send(&size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);

  if (size == 0)
    return;

  // Send actual data with non-blocking for potential overlap
  MPI_Request send_req;

  if (payload_size_ == 0) {
    // Zero-payload: send keys only
    std::vector<unsigned long> keys(size);
    for (size_t i = 0; i < size; ++i) {
      keys[i] = data[i].key;
    }

    MPI_Isend(keys.data(), size, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD,
              &send_req);
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
  } else {
    // Non-zero payload: pack and send complete records
    const size_t record_bytes = sizeof(unsigned long) + payload_size_;
    std::vector<char> buffer(size * record_bytes);
    char *ptr = buffer.data();

    // Pack all records
    for (const auto &rec : data) {
      memcpy(ptr, &rec.key, sizeof(unsigned long));
      ptr += sizeof(unsigned long);
      if (rec.payload && rec.payload_size > 0) {
        memcpy(ptr, rec.payload, rec.payload_size);
      }
      ptr += rec.payload_size;
    }

    MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
              &send_req);
    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
  }
}

/**
 * @brief FastFlow-parallelized k-way merge of multiple sorted partitions
 */
std::vector<Record> HybridMergeSort::parallel_k_way_merge(
    const std::vector<std::vector<Record>> &partitions) {
  // Filter out empty partitions
  std::vector<const std::vector<Record> *> non_empty_partitions;
  size_t total_elements = 0;

  for (const auto &partition : partitions) {
    if (!partition.empty()) {
      non_empty_partitions.push_back(&partition);
      total_elements += partition.size();
    }
  }

  if (non_empty_partitions.empty()) {
    return std::vector<Record>();
  }

  if (non_empty_partitions.size() == 1) {
    // Only one non-empty partition, just copy it
    std::vector<Record> result;
    result.reserve(non_empty_partitions[0]->size());
    for (const auto &rec : *non_empty_partitions[0]) {
      result.push_back(std::move(const_cast<Record &>(rec)));
    }
    return result;
  }

  // For large datasets with multiple partitions, use FastFlow farm for parallel
  // k-way merge
  if (total_elements > config_.min_local_threshold &&
      config_.parallel_threads > 1 && non_empty_partitions.size() > 2) {
    return fastflow_parallel_k_way_merge(non_empty_partitions, total_elements);
  } else {
    // For smaller datasets, use efficient sequential k-way merge
    return sequential_k_way_merge(non_empty_partitions, total_elements);
  }
}

/**
 * @brief Efficient sequential k-way merge using heap-based priority queue
 */
std::vector<Record> HybridMergeSort::sequential_k_way_merge(
    const std::vector<const std::vector<Record> *> &partitions,
    size_t total_elements) {
  // Stream state for each partition
  struct StreamState {
    const std::vector<Record> *partition;
    size_t index;

    StreamState(const std::vector<Record> *p, size_t i)
        : partition(p), index(i) {}

    bool has_more() const { return index < partition->size(); }
    const Record &current() const { return (*partition)[index]; }
    void advance() { ++index; }

    // Priority queue comparison (min-heap based on key)
    bool operator>(const StreamState &other) const {
      return current().key > other.current().key;
    }
  };

  // Initialize priority queue with first element from each non-empty partition
  std::priority_queue<StreamState, std::vector<StreamState>,
                      std::greater<StreamState>>
      heap;

  for (const auto *partition : partitions) {
    if (!partition->empty()) {
      heap.emplace(partition, 0);
    }
  }

  // Result vector with pre-allocated capacity
  std::vector<Record> result;
  result.reserve(total_elements);

  // K-way merge using heap
  while (!heap.empty()) {
    // Get minimum element
    StreamState current = heap.top();
    heap.pop();

    // Move the minimum record to result
    result.push_back(std::move(const_cast<Record &>(current.current())));

    // Advance stream and re-insert if more elements available
    current.advance();
    if (current.has_more()) {
      heap.push(current);
    }
  }

  return result;
}

/**
 * @brief FastFlow farm-based parallel k-way merge for large datasets
 */
std::vector<Record> HybridMergeSort::fastflow_parallel_k_way_merge(
    const std::vector<const std::vector<Record> *> &partitions,
    size_t total_elements) {
  // For very large datasets, we can implement parallel k-way merge
  // Strategy: Split partitions into groups and merge each group in parallel,
  // then do final merge of intermediate results

  const size_t num_partitions = partitions.size();
  const size_t workers = std::min(config_.parallel_threads, num_partitions);

  if (workers <= 1 || num_partitions <= 2) {
    // Fall back to sequential merge
    return sequential_k_way_merge(partitions, total_elements);
  }

  // Phase 1: Parallel merge of partition groups
  std::vector<std::vector<Record>> intermediate_results(workers);
  std::vector<std::thread> merge_threads;

  // Distribute partitions among workers
  const size_t partitions_per_worker = num_partitions / workers;
  const size_t remainder = num_partitions % workers;

  size_t partition_offset = 0;
  for (size_t worker = 0; worker < workers; ++worker) {
    size_t worker_partitions =
        partitions_per_worker + (worker < remainder ? 1 : 0);

    if (worker_partitions > 0) {
      merge_threads.emplace_back([this, &partitions, &intermediate_results,
                                  worker, partition_offset,
                                  worker_partitions]() {
        // Create subset of partitions for this worker
        std::vector<const std::vector<Record> *> worker_partitions_subset;
        size_t worker_total_elements = 0;

        for (size_t i = 0; i < worker_partitions; ++i) {
          worker_partitions_subset.push_back(partitions[partition_offset + i]);
          worker_total_elements += partitions[partition_offset + i]->size();
        }

        // Sequential k-way merge for this worker's partitions
        intermediate_results[worker] = sequential_k_way_merge(
            worker_partitions_subset, worker_total_elements);
      });
    }

    partition_offset += worker_partitions;
  }

  // Wait for all parallel merges to complete
  for (auto &thread : merge_threads) {
    thread.join();
  }

  // Phase 2: Final merge of intermediate results
  std::vector<const std::vector<Record> *> intermediate_partitions;
  for (const auto &result : intermediate_results) {
    if (!result.empty()) {
      intermediate_partitions.push_back(&result);
    }
  }

  return sequential_k_way_merge(intermediate_partitions, total_elements);
}

/**
 * @brief Update performance metrics for execution phase (UNCHANGED)
 */
void HybridMergeSort::update_metrics(const std::string &phase,
                                     double elapsed_time) {
  if (phase == "local_sort")
    metrics_.local_sort_time = elapsed_time;
  else if (phase == "merge")
    metrics_.merge_time = elapsed_time;
  else if (phase == "distribution")
    metrics_.communication_time = elapsed_time;
}

} // namespace hybrid
