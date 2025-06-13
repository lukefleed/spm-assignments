/**
 * @file mpi_ff_mergesort.cpp
 * @brief Hybrid MPI+FastFlow distributed mergesort with
 * computation-communication overlap
 */

#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

// Forward declaration
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace {
// Helper functions local to this translation unit
// CORREZIONE: Il tipo corretto è std::vector<Record>, non
// std::vector<hybrid::Record>
void pack_records(const std::vector<Record> &records, std::vector<char> &buffer,
                  size_t payload_size) {
  char *ptr = buffer.data();
  for (const auto &rec : records) {
    memcpy(ptr, &rec.key, sizeof(unsigned long));
    ptr += sizeof(unsigned long);
    if (payload_size > 0 && rec.payload) {
      memcpy(ptr, rec.payload, payload_size);
    }
    ptr += payload_size;
  }
}

// CORREZIONE: Il tipo corretto è std::vector<Record>
void unpack_records(const char *buffer, size_t num_records,
                    std::vector<Record> &records, size_t payload_size) {
  records.clear();
  records.reserve(num_records);
  const char *ptr = buffer;
  for (size_t i = 0; i < num_records; ++i) {
    auto &rec = records.emplace_back(payload_size);
    memcpy(&rec.key, ptr, sizeof(unsigned long));
    ptr += sizeof(unsigned long);
    if (payload_size > 0 && rec.payload) {
      memcpy(rec.payload, ptr, payload_size);
    }
    ptr += payload_size;
  }
}

} // namespace

namespace hybrid {

struct MergeStep {
  int source_rank;
  std::vector<Record> buffer;
  std::unique_ptr<char[]> packed_buffer;
  MPI_Request request = MPI_REQUEST_NULL;
  bool data_received = false;

  MergeStep() = default;
  MergeStep(MergeStep &&other) noexcept
      : source_rank(other.source_rank), buffer(std::move(other.buffer)),
        packed_buffer(std::move(other.packed_buffer)), request(other.request),
        data_received(other.data_received) {
    other.request = MPI_REQUEST_NULL;
  }
  MergeStep &operator=(MergeStep &&other) noexcept {
    if (this != &other) {
      source_rank = other.source_rank;
      buffer = std::move(other.buffer);
      packed_buffer = std::move(other.packed_buffer);
      request = other.request;
      data_received = other.data_received;
      other.request = MPI_REQUEST_NULL;
    }
    return *this;
  }

  MergeStep(const MergeStep &) = delete;
  MergeStep &operator=(const MergeStep &) = delete;
};

HybridMergeSort::HybridMergeSort(const HybridConfig &config)
    : config_(config), mpi_rank_(-1), mpi_size_(-1), payload_size_(0),
      metrics_{} {
  // Verify MPI is already initialized
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error(
        "MPI must be initialized before constructing HybridMergeSort");
  }

  // Verify MPI threading support
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error("MPI does not support MPI_THREAD_FUNNELED");
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  if (config_.parallel_threads == 0) {
    throw std::invalid_argument(
        "parallel_threads must be explicitly set (> 0)");
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
  if (mpi_size_ > 1) {
    hierarchical_merge_with_overlap(local_data);
  }
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

  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);
  size_t base_count = total_num_records / mpi_size_;
  size_t remainder = total_num_records % mpi_size_;
  for (int i = 0; i < mpi_size_; ++i) {
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // CORREZIONE: Inizializza il vettore vuoto. Sarà popolato da unpack_records.
  local_data.clear();

  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  std::vector<int> send_counts_bytes(mpi_size_);
  std::vector<int> displs_bytes(mpi_size_);
  for (int i = 0; i < mpi_size_; ++i) {
    send_counts_bytes[i] = send_counts[i] * record_byte_size;
    displs_bytes[i] = displs[i] * record_byte_size;
  }

  std::vector<char> send_buffer;
  if (mpi_rank_ == 0) {
    send_buffer.resize(total_num_records * record_byte_size);
    pack_records(global_data, send_buffer, payload_size_);
  }

  std::vector<char> recv_buffer(send_counts_bytes[mpi_rank_]);
  MPI_Scatterv(send_buffer.data(), send_counts_bytes.data(),
               displs_bytes.data(), MPI_BYTE, recv_buffer.data(),
               recv_buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

  unpack_records(recv_buffer.data(), send_counts[mpi_rank_], local_data,
                 payload_size_);
  metrics_.bytes_communicated += recv_buffer.size();
}

void HybridMergeSort::sort_local_data(std::vector<Record> &data) {
  if (data.empty())
    return;
  if (data.size() >= config_.min_local_threshold &&
      config_.parallel_threads > 1) {
    parallel_mergesort(data, config_.parallel_threads);
  } else {
    std::sort(data.begin(), data.end());
  }
}

void HybridMergeSort::hierarchical_merge_with_overlap(
    std::vector<Record> &local_data) {
  std::vector<MergeStep> steps_to_process;
  int total_receives_posted = 0;

  for (int step = 1; step < mpi_size_; step *= 2) {
    if ((mpi_rank_ % (2 * step)) != 0) {
      int target = mpi_rank_ - step;
      send_data_and_exit(local_data, target);
      local_data.clear();
      return;
    }

    int source = mpi_rank_ + step;
    if (source < mpi_size_) {
      steps_to_process.emplace_back();
      MergeStep &current_step = steps_to_process.back();
      current_step.source_rank = source;

      size_t incoming_size;
      MPI_Recv(&incoming_size, 1, MPI_UNSIGNED_LONG, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      if (incoming_size > 0) {
        const size_t buffer_bytes =
            incoming_size * (sizeof(unsigned long) + payload_size_);
        current_step.packed_buffer = std::make_unique<char[]>(buffer_bytes);
        current_step.buffer.reserve(incoming_size);

        MPI_Irecv(current_step.packed_buffer.get(), buffer_bytes, MPI_BYTE,
                  source, 1, MPI_COMM_WORLD, &current_step.request);
        total_receives_posted++;
        metrics_.bytes_communicated += buffer_bytes;
      } else {
        current_step.data_received = true;
      }
    }
  }

  int completed_merges = 0;
  while (completed_merges < total_receives_posted) {
    std::vector<MPI_Request> pending_requests;
    std::vector<int> request_map;

    for (size_t i = 0; i < steps_to_process.size(); ++i) {
      if (!steps_to_process[i].data_received) {
        pending_requests.push_back(steps_to_process[i].request);
        request_map.push_back(i);
      }
    }

    if (pending_requests.empty())
      break;

    int completed_idx = MPI_UNDEFINED;
    MPI_Waitany(pending_requests.size(), pending_requests.data(),
                &completed_idx, MPI_STATUS_IGNORE);

    if (completed_idx != MPI_UNDEFINED) {
      int original_step_idx = request_map[completed_idx];
      MergeStep &step = steps_to_process[original_step_idx];
      step.data_received = true;
      completed_merges++;

      unpack_records(step.packed_buffer.get(), step.buffer.capacity(),
                     step.buffer, payload_size_);

      merge_two_sorted_arrays(local_data, step.buffer);
    }
  }
}

void HybridMergeSort::send_data_and_exit(const std::vector<Record> &data,
                                         int target) {
  size_t size = data.size();
  MPI_Send(&size, 1, MPI_UNSIGNED_LONG, target, 0, MPI_COMM_WORLD);

  if (size > 0) {
    std::vector<char> send_buffer(size *
                                  (sizeof(unsigned long) + payload_size_));
    pack_records(data, send_buffer, payload_size_);
    MPI_Send(send_buffer.data(), send_buffer.size(), MPI_BYTE, target, 1,
             MPI_COMM_WORLD);
  }
}

void HybridMergeSort::merge_two_sorted_arrays(
    std::vector<Record> &local_data, std::vector<Record> &partner_data) {
  if (partner_data.empty())
    return;
  if (local_data.empty()) {
    local_data = std::move(partner_data);
    return;
  }
  std::vector<Record> merged;
  merged.reserve(local_data.size() + partner_data.size());
  auto it1 = std::make_move_iterator(local_data.begin());
  auto end1 = std::make_move_iterator(local_data.end());
  auto it2 = std::make_move_iterator(partner_data.begin());
  auto end2 = std::make_move_iterator(partner_data.end());
  std::merge(it1, end1, it2, end2, std::back_inserter(merged));
  local_data = std::move(merged);
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
