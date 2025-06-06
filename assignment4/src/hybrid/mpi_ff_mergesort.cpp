#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../fastflow/ff_mergesort.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

// Forward declaration of FastFlow function
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads);

namespace hybrid {

// ============================================================================
// HybridMergeSort Implementation
// ============================================================================

HybridMergeSort::HybridMergeSort(const HybridConfig &config) : config_(config) {
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error(
        "MPI implementation does not support required threading level");
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  if (config_.ff_threads == 0) {
    config_.ff_threads = utils::get_optimal_ff_threads();
  }

  // Reset metrics
  metrics_ = HybridMetrics{};
}

HybridMergeSort::~HybridMergeSort() {
  // Cleanup handled by smart pointers
}

std::vector<Record> HybridMergeSort::sort(std::vector<Record> &data,
                                          size_t payload_size) {
  Timer total_timer;

  // Initialize MPI datatype and communication manager
  datatype_ = std::make_unique<mpi_comm::RecordDatatype>(payload_size);
  comm_manager_ = std::make_unique<mpi_comm::AsyncCommManager>(*datatype_);

  std::vector<Record> local_data;

  // Phase 1: Data Distribution
  {
    Timer dist_timer;
    distribute_data(local_data, data);
    update_metrics("distribution", dist_timer.elapsed_ms());
  }

  // Phase 2: Local FastFlow Sorting
  {
    Timer sort_timer;
    sort_local_data(local_data);
    update_metrics("local_sort", sort_timer.elapsed_ms());
  }

  // Phase 3: Load Balancing (if enabled)
  if (config_.load_balance_factor > 0.0) {
    Timer balance_timer;
    balance_load(local_data);
    update_metrics("load_balance", balance_timer.elapsed_ms());
  }

  // Phase 4: Hierarchical Distributed Merging
  {
    Timer merge_timer;
    hierarchical_merge(local_data);
    update_metrics("merge", merge_timer.elapsed_ms());
  }

  // Phase 5: Final Result Gathering
  std::vector<Record> final_result;
  {
    Timer gather_timer;
    gather_final_result(final_result, local_data);
    update_metrics("gather", gather_timer.elapsed_ms());
  }

  metrics_.total_time = total_timer.elapsed_ms();
  metrics_.local_elements = local_data.size();

  return final_result;
}

void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  // First, broadcast the total data size from root to all processes
  size_t total_size = global_data.size();
  MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  mpi_comm::DataDistributor distributor(mpi_rank_, mpi_size_, total_size,
                                        datatype_->payload_size());

  size_t local_size = distributor.local_size();
  local_data.reserve(local_size);

  // Calculate send counts and displacements for scatterv
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> send_displs(mpi_size_);

  for (int i = 0; i < mpi_size_; ++i) {
    send_counts[i] = static_cast<int>(distributor.size_for_rank(i));
    send_displs[i] = static_cast<int>(distributor.start_for_rank(i));
  }

  // Use optimized scatter operation - now all ranks have the counts
  mpi_comm::collective::scatter_records(global_data, local_data, send_counts, 0,
                                        *datatype_, MPI_COMM_WORLD);

  // Update communication metrics
  size_t bytes_transferred =
      local_size * (sizeof(unsigned long) + datatype_->payload_size());
  metrics_.bytes_communicated += bytes_transferred;
}

void HybridMergeSort::sort_local_data(std::vector<Record> &local_data) {
  if (local_data.empty()) {
    return;
  }

  // Use FastFlow for local sorting if data size is above threshold
  if (local_data.size() >= config_.min_local_threshold &&
      config_.ff_threads > 1) {
    ff_pipeline_two_farms_mergesort(local_data, config_.ff_threads);
  } else {
    std::sort(local_data.begin(), local_data.end());
  }
}

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  mpi_comm::MergeTree merge_tree(mpi_rank_, mpi_size_);

  // Track which processes are still active
  bool still_active = true;

  for (int level = 0; level < merge_tree.height() && still_active; ++level) {
    if (!merge_tree.is_active_at_level(level)) {
      still_active = false;
      break;
    }

    int partner = merge_tree.get_partner(level);
    bool is_receiver = merge_tree.is_receiver(level);

    if (partner < mpi_size_) {
      auto result = merge_with_partner(local_data, partner, is_receiver);

      if (is_receiver) {
        local_data = std::move(result);

        // Update communication metrics
        size_t bytes_transferred =
            local_data.size() *
            (sizeof(unsigned long) + datatype_->payload_size());
        metrics_.bytes_communicated += bytes_transferred;
      } else {
        // Sender becomes inactive
        local_data.clear();
        still_active = false;
      }
    }

    // All processes must participate in MPI_Comm_split
    int color = still_active ? 1 : MPI_UNDEFINED;
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank_, &active_comm);

    if (still_active && active_comm != MPI_COMM_NULL) {
      MPI_Barrier(active_comm);
      MPI_Comm_free(&active_comm);
    }
  }
}

void HybridMergeSort::balance_load(std::vector<Record> &local_data) {
  // Gather local sizes from all processes
  std::vector<size_t> all_sizes(mpi_size_);
  size_t local_size = local_data.size();

  MPI_Allgather(&local_size, 1, MPI_UNSIGNED_LONG, all_sizes.data(), 1,
                MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  // Calculate load imbalance
  size_t min_size = *std::min_element(all_sizes.begin(), all_sizes.end());
  size_t max_size = *std::max_element(all_sizes.begin(), all_sizes.end());

  if (min_size == 0 || mpi_size_ == 1)
    return;

  double imbalance_ratio = static_cast<double>(max_size) / min_size - 1.0;
  metrics_.load_balance_ratio = 1.0 / (1.0 + imbalance_ratio);

  // Perform redistribution if imbalance exceeds threshold
  if (imbalance_ratio > config_.load_balance_factor) {
    // Calculate total elements
    size_t total_elements = 0;
    for (size_t s : all_sizes) {
      total_elements += s;
    }

    // Calculate target distribution
    size_t base_size = total_elements / mpi_size_;
    size_t remainder = total_elements % mpi_size_;

    std::vector<size_t> target_sizes(mpi_size_);
    for (int i = 0; i < mpi_size_; ++i) {
      target_sizes[i] = base_size + (i < static_cast<int>(remainder) ? 1 : 0);
    }

    // Calculate send/recv counts using MPI_Alltoall
    std::vector<int> send_counts(mpi_size_, 0);
    std::vector<int> recv_counts(mpi_size_, 0);

    // First, each process announces how much it wants to send to others
    std::vector<int> send_requests(mpi_size_, 0);
    std::vector<int> recv_requests(mpi_size_, 0);

    // Calculate excess/deficit
    int my_excess = static_cast<int>(local_size) -
                    static_cast<int>(target_sizes[mpi_rank_]);

    if (my_excess > 0) {
      // Distribute excess to processes with deficit
      for (int i = 0; i < mpi_size_ && my_excess > 0; ++i) {
        if (i != mpi_rank_) {
          int their_deficit = static_cast<int>(target_sizes[i]) -
                              static_cast<int>(all_sizes[i]);
          if (their_deficit > 0) {
            int to_send = std::min(my_excess, their_deficit);
            send_requests[i] = to_send;
            my_excess -= to_send;
          }
        }
      }
    }

    // Exchange send/recv requests
    MPI_Alltoall(send_requests.data(), 1, MPI_INT, recv_requests.data(), 1,
                 MPI_INT, MPI_COMM_WORLD);

    // Set up actual send/recv counts
    send_counts = send_requests;
    recv_counts = recv_requests;

    // Calculate displacements
    std::vector<int> send_displs(mpi_size_, 0);
    std::vector<int> recv_displs(mpi_size_, 0);

    int send_offset = 0, recv_offset = 0;
    for (int i = 0; i < mpi_size_; ++i) {
      send_displs[i] = send_offset;
      recv_displs[i] = recv_offset;
      send_offset += send_counts[i];
      recv_offset += recv_counts[i];
    }

    // Execute redistribution
    std::vector<Record> redistributed_data;
    mpi_comm::efficient_alltoallv(local_data, send_counts, send_displs,
                                  redistributed_data, recv_counts, recv_displs,
                                  *datatype_, MPI_COMM_WORLD);

    // Keep the data we're not sending and merge with received data
    std::vector<Record> kept_data;
    size_t keep_count = local_data.size() - send_offset;
    if (keep_count > 0) {
      kept_data.reserve(keep_count);
      for (size_t i = 0; i < keep_count; ++i) {
        kept_data.emplace_back(std::move(local_data[i]));
      }
    }

    // Merge kept and received data
    local_data = merge_sorted_vectors(kept_data, redistributed_data);

    // Update metrics
    size_t bytes_redistributed =
        (send_offset + recv_offset) *
        (sizeof(unsigned long) + datatype_->payload_size());
    metrics_.bytes_communicated += bytes_redistributed;
  }
}

std::vector<Record>
HybridMergeSort::merge_with_partner(const std::vector<Record> &local_data,
                                    int partner_rank, bool is_receiver) {
  if (is_receiver) {
    // Receive data from partner and merge
    size_t partner_size;
    MPI_Recv(&partner_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (config_.use_async_merging) {
      // Use non-blocking communication for overlap
      size_t recv_id = comm_manager_->async_recv(partner_size, partner_rank, 1);

      // Wait for receive completion
      std::vector<Record> partner_data = comm_manager_->wait_and_get(recv_id);

      // Merge the two sorted sequences
      return merge_sorted_vectors(local_data, partner_data);
    } else {
      // Blocking communication
      std::vector<Record> partner_data(partner_size);
      for (auto &record : partner_data) {
        if (datatype_->payload_size() > 0) {
          record.payload = new char[datatype_->payload_size()];
          record.payload_size = datatype_->payload_size();
        }
      }

      MPI_Recv(partner_data.data(), partner_size, datatype_->get(),
               partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Merge the two sorted sequences
      return merge_sorted_vectors(local_data, partner_data);
    }
  } else {
    // Send data to partner (sender becomes inactive)
    size_t local_size = local_data.size();
    MPI_Send(&local_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0,
             MPI_COMM_WORLD);

    if (config_.use_async_merging) {
      comm_manager_->async_send(local_data, partner_rank, 1);
      comm_manager_->wait_all();
    } else {
      MPI_Send(local_data.data(), local_size, datatype_->get(), partner_rank, 1,
               MPI_COMM_WORLD);
    }

    return {}; // Sender becomes inactive
  }
}

void HybridMergeSort::gather_final_result(
    std::vector<Record> &result_data, const std::vector<Record> &local_data) {
  // Only rank 0 should have data at this point
  if (mpi_rank_ == 0) {
    result_data = std::move(const_cast<std::vector<Record> &>(local_data));
  } else {
    result_data.clear();
  }
}

void HybridMergeSort::update_metrics(const std::string &phase,
                                     double elapsed_time,
                                     size_t bytes_transferred) {
  if (phase == "local_sort") {
    metrics_.local_sort_time += elapsed_time;
  } else if (phase == "merge") {
    metrics_.merge_time += elapsed_time;
  } else if (phase == "distribution" || phase == "gather" ||
             phase == "load_balance") {
    metrics_.communication_time += elapsed_time;
  }

  metrics_.bytes_communicated += bytes_transferred;
}

bool HybridMergeSort::validate_result(
    const std::vector<Record> &sorted_data,
    const std::vector<Record> &original_data) const {
  if (mpi_rank_ != 0)
    return true; // Only validate on root

  if (sorted_data.size() != original_data.size())
    return false;

  // Check if sorted
  for (size_t i = 1; i < sorted_data.size(); ++i) {
    if (sorted_data[i] < sorted_data[i - 1])
      return false;
  }

  // Check if permutation (simplified key-only check)
  std::vector<unsigned long> orig_keys, sorted_keys;
  for (const auto &record : original_data)
    orig_keys.push_back(record.key);
  for (const auto &record : sorted_data)
    sorted_keys.push_back(record.key);

  std::sort(orig_keys.begin(), orig_keys.end());
  std::sort(sorted_keys.begin(), sorted_keys.end());

  return orig_keys == sorted_keys;
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

namespace utils {

size_t get_optimal_ff_threads() {
  size_t hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 4; // Fallback

  // Use 75% of available cores to leave room for MPI communication
  return std::max(size_t(1), static_cast<size_t>(hw_threads * 0.75));
}

double estimate_comm_cost(size_t data_size, size_t payload_size,
                          int num_processes) {
  double bytes_per_record = sizeof(unsigned long) + payload_size;
  double total_bytes = data_size * bytes_per_record;

  // Simplified cost model: linear in data size and log in processes
  double base_cost = total_bytes / (1024.0 * 1024.0); // MB
  double scaling_factor = std::log2(num_processes);

  return base_cost * scaling_factor;
}

std::vector<size_t> calculate_optimal_distribution(size_t total_elements,
                                                   int num_processes,
                                                   double load_factor) {
  std::vector<size_t> distribution(num_processes);
  size_t base_size = total_elements / num_processes;
  size_t remainder = total_elements % num_processes;

  for (int i = 0; i < num_processes; ++i) {
    distribution[i] = base_size + (i < static_cast<int>(remainder) ? 1 : 0);
  }

  // Apply load balancing factor (simplified)
  if (load_factor > 0.0 && num_processes > 1) {
    size_t adjustment = static_cast<size_t>(base_size * load_factor);
    for (int i = 1; i < num_processes; i += 2) {
      if (distribution[i - 1] > adjustment) {
        distribution[i - 1] -= adjustment;
        distribution[i] += adjustment;
      }
    }
  }

  return distribution;
}

bool validate_mpi_environment() {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized)
    return false;

  int provided;
  MPI_Query_thread(&provided);
  return provided >= MPI_THREAD_FUNNELED;
}

double benchmark_fastflow_performance(size_t test_size, size_t payload_size,
                                      size_t num_threads) {
  // Create test data
  std::vector<Record> test_data;
  test_data.reserve(test_size);

  for (size_t i = 0; i < test_size; ++i) {
    test_data.emplace_back(payload_size);
    test_data.back().key = test_size - i; // Reverse order for worst case
  }

  Timer timer;
  ff_pipeline_two_farms_mergesort(test_data, num_threads);
  return timer.elapsed_ms();
}

} // namespace utils

// ============================================================================
// High-level Convenience Function
// ============================================================================

std::vector<Record> hybrid_mergesort(std::vector<Record> &data,
                                     size_t payload_size, size_t ff_threads) {
  if (!utils::validate_mpi_environment()) {
    throw std::runtime_error("Invalid MPI environment for hybrid mergesort");
  }

  HybridConfig config;
  config.ff_threads =
      (ff_threads == 0) ? utils::get_optimal_ff_threads() : ff_threads;

  // Auto-tune configuration based on problem size
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (data.size() < config.ff_threads * 1024 * mpi_size) {
    // Small problem: disable some optimizations
    config.enable_overlap = false;
    config.use_async_merging = false;
    config.load_balance_factor = 0.0;
  } else if (data.size() > config.ff_threads * 1024 * 1024 * mpi_size) {
    // Large problem: enable all optimizations
    config.enable_overlap = true;
    config.use_async_merging = true;
    config.load_balance_factor = 0.15;
  }

  HybridMergeSort sorter(config);
  return sorter.sort(data, payload_size);
}

/**
 * @brief Helper function to merge two sorted vectors of Records
 * Uses move semantics to avoid copy constructor issues
 */
std::vector<Record>
HybridMergeSort::merge_sorted_vectors(const std::vector<Record> &left,
                                      std::vector<Record> &right) {
  std::vector<Record> result;
  result.reserve(left.size() + right.size());

  size_t i = 0, j = 0;

  // Merge by moving elements to avoid copy issues
  while (i < left.size() && j < right.size()) {
    if (left[i].key <= right[j].key) {
      // Create a new Record and move from left
      result.emplace_back(left[i].payload_size);
      result.back().key = left[i].key;
      if (left[i].payload_size > 0 && left[i].payload) {
        std::memcpy(result.back().payload, left[i].payload,
                    left[i].payload_size);
      }
      ++i;
    } else {
      // Move directly from right (which is non-const)
      result.emplace_back(std::move(right[j]));
      ++j;
    }
  }

  // Add remaining elements from left
  while (i < left.size()) {
    result.emplace_back(left[i].payload_size);
    result.back().key = left[i].key;
    if (left[i].payload_size > 0 && left[i].payload) {
      std::memcpy(result.back().payload, left[i].payload, left[i].payload_size);
    }
    ++i;
  }

  // Add remaining elements from right
  while (j < right.size()) {
    result.emplace_back(std::move(right[j]));
    ++j;
  }

  return result;
}

} // namespace hybrid
