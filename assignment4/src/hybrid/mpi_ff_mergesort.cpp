// FILE: src/hybrid/mpi_ff_mergesort.cpp

#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// External function from FastFlow implementation
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace hybrid {

HybridMergeSort::HybridMergeSort(const HybridConfig &config)
    : config_(config), mpi_rank_(-1), mpi_size_(-1), payload_size_(0),
      metrics_{} {
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error("MPI does not support MPI_THREAD_FUNNELED.");
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
  if (config_.parallel_threads == 0) {
    config_.parallel_threads = utils::get_optimal_parallel_threads();
  }
}

HybridMergeSort::~HybridMergeSort() = default;

std::vector<Record> HybridMergeSort::sort(std::vector<Record> &data,
                                          size_t payload_size) {
  Timer total_timer;
  payload_size_ = payload_size;
  std::vector<Record> local_data;

  Timer dist_timer;
  distribute_data(local_data, data);
  update_metrics("distribution", dist_timer.elapsed_ms());

  Timer sort_timer;
  sort_local_data(local_data);
  update_metrics("local_sort", sort_timer.elapsed_ms());

  Timer merge_timer;
  hierarchical_merge(local_data);
  update_metrics("merge", merge_timer.elapsed_ms());

  metrics_.total_time = total_timer.elapsed_ms();
  metrics_.local_elements = (mpi_rank_ == 0) ? local_data.size() : 0;

  return local_data;
}

void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  size_t total_num_records = (mpi_rank_ == 0) ? global_data.size() : 0;
  MPI_Bcast(&total_num_records, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (total_num_records == 0)
    return;

  // Calculate distribution
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);

  size_t base_count = total_num_records / mpi_size_;
  size_t remainder = total_num_records % mpi_size_;

  for (int i = 0; i < mpi_size_; ++i) {
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // Pre-allocate and construct records
  local_data.clear();
  local_data.reserve(send_counts[mpi_rank_]);
  for (int i = 0; i < send_counts[mpi_rank_]; ++i) {
    local_data.emplace_back(payload_size_);
  }

  if (payload_size_ == 0) {
    // Optimize for zero payload - just scatter keys
    std::vector<unsigned long> keys;
    if (mpi_rank_ == 0) {
      keys.reserve(global_data.size());
      for (const auto &rec : global_data) {
        keys.push_back(rec.key);
      }
    }

    std::vector<unsigned long> local_keys(send_counts[mpi_rank_]);
    MPI_Scatterv(keys.data(), send_counts.data(), displs.data(),
                 MPI_UNSIGNED_LONG, local_keys.data(), send_counts[mpi_rank_],
                 MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < local_keys.size(); ++i) {
      local_data[i].key = local_keys[i];
    }
  } else {
    // For non-zero payload, use single buffer approach
    const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
    std::vector<int> send_counts_bytes(mpi_size_);
    std::vector<int> displs_bytes(mpi_size_);

    for (int i = 0; i < mpi_size_; ++i) {
      send_counts_bytes[i] = send_counts[i] * record_byte_size;
      displs_bytes[i] = displs[i] * record_byte_size;
    }

    // Pack only once on root
    std::vector<char> send_buffer;
    if (mpi_rank_ == 0) {
      send_buffer.resize(total_num_records * record_byte_size);
      char *ptr = send_buffer.data();
      for (const auto &rec : global_data) {
        memcpy(ptr, &rec.key, sizeof(unsigned long));
        ptr += sizeof(unsigned long);
        if (rec.payload && rec.payload_size > 0) {
          memcpy(ptr, rec.payload, rec.payload_size);
        }
        ptr += rec.payload_size;
      }
    }

    std::vector<char> recv_buffer(send_counts_bytes[mpi_rank_]);
    MPI_Scatterv(send_buffer.data(), send_counts_bytes.data(),
                 displs_bytes.data(), MPI_BYTE, recv_buffer.data(),
                 recv_buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Unpack directly into pre-allocated local_data
    const char *ptr = recv_buffer.data();
    for (int i = 0; i < send_counts[mpi_rank_]; ++i) {
      memcpy(&local_data[i].key, ptr, sizeof(unsigned long));
      ptr += sizeof(unsigned long);
      if (payload_size_ > 0) {
        memcpy(local_data[i].payload, ptr, payload_size_);
      }
      ptr += payload_size_;
    }
  }

  metrics_.bytes_communicated +=
      send_counts[mpi_rank_] * (sizeof(unsigned long) + payload_size_);
}

void HybridMergeSort::sort_local_data(std::vector<Record> &data) {
  if (data.empty())
    return;

  // Use existing parallel_mergesort implementation
  if (data.size() >= config_.min_local_threshold &&
      config_.parallel_threads > 1) {
    parallel_mergesort(data, config_.parallel_threads);
  } else {
    std::sort(data.begin(), data.end());
  }
}

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  // Use binary tree reduction instead of comm splitting
  for (int step = 1; step < mpi_size_; step *= 2) {
    if ((mpi_rank_ % (2 * step)) == 0) {
      int source = mpi_rank_ + step;
      if (source < mpi_size_) {
        // Receive size first
        size_t incoming_size;
        MPI_Recv(&incoming_size, 1, MPI_UNSIGNED_LONG, source, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (incoming_size > 0) {
          // Pre-allocate partner data with proper construction
          std::vector<Record> partner_data;
          partner_data.reserve(incoming_size);
          for (size_t i = 0; i < incoming_size; ++i) {
            partner_data.emplace_back(payload_size_);
          }

          if (payload_size_ == 0) {
            // Optimize for zero payload
            std::vector<unsigned long> keys(incoming_size);
            MPI_Recv(keys.data(), incoming_size, MPI_UNSIGNED_LONG, source, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t i = 0; i < incoming_size; ++i) {
              partner_data[i].key = keys[i];
            }
          } else {
            // Use contiguous buffer for better MPI performance
            const size_t record_bytes = sizeof(unsigned long) + payload_size_;
            std::vector<char> buffer(incoming_size * record_bytes);

            MPI_Recv(buffer.data(), buffer.size(), MPI_BYTE, source, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Unpack
            const char *ptr = buffer.data();
            for (size_t i = 0; i < incoming_size; ++i) {
              memcpy(&partner_data[i].key, ptr, sizeof(unsigned long));
              ptr += sizeof(unsigned long);
              if (payload_size_ > 0) {
                memcpy(partner_data[i].payload, ptr, payload_size_);
              }
              ptr += payload_size_;
            }
          }

          // Merge using move semantics
          if (local_data.empty()) {
            local_data = std::move(partner_data);
          } else {
            std::vector<Record> merged;
            merged.reserve(local_data.size() + partner_data.size());

            // Manual merge with move semantics
            size_t i = 0, j = 0;
            while (i < local_data.size() && j < partner_data.size()) {
              if (local_data[i] < partner_data[j]) {
                merged.push_back(std::move(local_data[i++]));
              } else {
                merged.push_back(std::move(partner_data[j++]));
              }
            }

            // Move remaining elements
            while (i < local_data.size()) {
              merged.push_back(std::move(local_data[i++]));
            }
            while (j < partner_data.size()) {
              merged.push_back(std::move(partner_data[j++]));
            }

            local_data = std::move(merged);
          }

          metrics_.bytes_communicated +=
              incoming_size * (sizeof(unsigned long) + payload_size_);
        }
      }
    } else if ((mpi_rank_ % (2 * step)) == step) {
      int target = mpi_rank_ - step;

      // Send size first
      size_t size = local_data.size();
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG, target, 0, MPI_COMM_WORLD);

      if (size > 0) {
        if (payload_size_ == 0) {
          // Optimize for zero payload
          std::vector<unsigned long> keys(size);
          for (size_t i = 0; i < size; ++i) {
            keys[i] = local_data[i].key;
          }
          MPI_Send(keys.data(), size, MPI_UNSIGNED_LONG, target, 1,
                   MPI_COMM_WORLD);
        } else {
          // Pack once and send
          const size_t record_bytes = sizeof(unsigned long) + payload_size_;
          std::vector<char> buffer(size * record_bytes);
          char *ptr = buffer.data();

          for (const auto &rec : local_data) {
            memcpy(ptr, &rec.key, sizeof(unsigned long));
            ptr += sizeof(unsigned long);
            if (rec.payload && rec.payload_size > 0) {
              memcpy(ptr, rec.payload, rec.payload_size);
            }
            ptr += rec.payload_size;
          }

          MPI_Send(buffer.data(), buffer.size(), MPI_BYTE, target, 1,
                   MPI_COMM_WORLD);
        }
      }

      local_data.clear();
      break;
    }
  }
}

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
