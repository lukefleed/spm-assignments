// FILE: src/hybrid/mpi_ff_mergesort.cpp

#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads);

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
  if (config_.ff_threads == 0) {
    config_.ff_threads = utils::get_optimal_ff_threads();
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

  return local_data; // Only rank 0 has the final data, others have an empty
                     // vector
}

void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  size_t total_num_records = (mpi_rank_ == 0) ? global_data.size() : 0;
  MPI_Bcast(&total_num_records, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (total_num_records == 0)
    return;

  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  std::vector<int> send_counts_bytes(mpi_size_);
  std::vector<int> displs_bytes(mpi_size_);

  size_t offset_bytes = 0;
  for (int i = 0; i < mpi_size_; ++i) {
    size_t num_records_for_rank =
        total_num_records / mpi_size_ +
        (i < static_cast<int>(total_num_records % mpi_size_) ? 1 : 0);
    send_counts_bytes[i] = num_records_for_rank * record_byte_size;
    displs_bytes[i] = offset_bytes;
    offset_bytes += send_counts_bytes[i];
  }

  std::vector<char> send_buffer;
  if (mpi_rank_ == 0) {
    send_buffer = pack_records(global_data, record_byte_size);
  }

  std::vector<char> recv_buffer(send_counts_bytes[mpi_rank_]);
  MPI_Scatterv(send_buffer.data(), send_counts_bytes.data(),
               displs_bytes.data(), MPI_BYTE, recv_buffer.data(),
               recv_buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

  local_data = unpack_records(recv_buffer, payload_size_, record_byte_size);
  metrics_.bytes_communicated += recv_buffer.size();
}

void HybridMergeSort::sort_local_data(std::vector<Record> &data) {
  if (data.empty())
    return;
  if (data.size() >= config_.min_local_threshold && config_.ff_threads > 1) {
    ff_pipeline_two_farms_mergesort(data, config_.ff_threads);
  } else {
    std::sort(data.begin(), data.end());
  }
}

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  MPI_Comm active_comm = MPI_COMM_WORLD;
  int current_rank = mpi_rank_;
  int current_size = mpi_size_;

  while (current_size > 1) {
    int partner = -1;
    bool is_receiver = false;

    if ((current_rank % 2) == 0) {
      partner = current_rank + 1;
      if (partner < current_size)
        is_receiver = true;
    } else {
      partner = current_rank - 1;
    }

    if (partner != -1 && partner < current_size) {
      local_data =
          merge_with_partner(local_data, partner, is_receiver, active_comm);
    }

    int color =
        ((current_rank % 2 == 0) && (partner < current_size || partner == -1))
            ? 0
            : MPI_UNDEFINED;

    MPI_Comm next_comm;
    MPI_Comm_split(active_comm, color, current_rank, &next_comm);

    if (active_comm != MPI_COMM_WORLD)
      MPI_Comm_free(&active_comm);
    active_comm = next_comm;

    if (active_comm == MPI_COMM_NULL)
      break;

    MPI_Comm_rank(active_comm, &current_rank);
    MPI_Comm_size(active_comm, &current_size);
  }
  if (active_comm != MPI_COMM_WORLD && active_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&active_comm);
  }
}

std::vector<Record>
HybridMergeSort::merge_with_partner(std::vector<Record> &local_data,
                                    int partner_rank, bool is_receiver,
                                    MPI_Comm comm) {
  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  if (is_receiver) {
    size_t partner_byte_size = 0;
    MPI_Recv(&partner_byte_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0, comm,
             MPI_STATUS_IGNORE);
    if (partner_byte_size == 0)
      return std::move(local_data);

    std::vector<char> partner_buffer(partner_byte_size);
    MPI_Recv(partner_buffer.data(), partner_buffer.size(), MPI_BYTE,
             partner_rank, 1, comm, MPI_STATUS_IGNORE);

    metrics_.bytes_communicated += partner_buffer.size();

    std::vector<Record> partner_data =
        unpack_records(partner_buffer, payload_size_, record_byte_size);
    return merge_sorted_vectors(local_data, partner_data);
  } else {
    std::vector<char> send_buffer = pack_records(local_data, record_byte_size);
    size_t local_byte_size = send_buffer.size();
    MPI_Send(&local_byte_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0, comm);
    if (local_byte_size > 0) {
      MPI_Send(send_buffer.data(), send_buffer.size(), MPI_BYTE, partner_rank,
               1, comm);
    }
    return {};
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

std::vector<char>
HybridMergeSort::pack_records(const std::vector<Record> &records,
                              size_t record_byte_size) {
  std::vector<char> buffer(records.size() * record_byte_size);
  char *current = buffer.data();
  for (const auto &rec : records) {
    memcpy(current, &rec.key, sizeof(unsigned long));
    current += sizeof(unsigned long);
    if (rec.payload_size > 0 && rec.payload != nullptr) {
      memcpy(current, rec.payload, rec.payload_size);
    }
    current += rec.payload_size;
  }
  return buffer;
}

std::vector<Record>
HybridMergeSort::unpack_records(const std::vector<char> &buffer,
                                size_t payload_size, size_t record_byte_size) {
  if (buffer.empty())
    return {};
  size_t num_records = buffer.size() / record_byte_size;
  std::vector<Record> records;
  records.reserve(num_records);
  const char *current = buffer.data();
  for (size_t i = 0; i < num_records; ++i) {
    Record rec(payload_size);
    memcpy(&rec.key, current, sizeof(unsigned long));
    current += sizeof(unsigned long);
    if (payload_size > 0) {
      memcpy(rec.payload, current, payload_size);
    }
    current += payload_size;
    records.push_back(std::move(rec));
  }
  return records;
}

std::vector<Record>
HybridMergeSort::merge_sorted_vectors(std::vector<Record> &left,
                                      std::vector<Record> &right) {
  std::vector<Record> result;
  result.reserve(left.size() + right.size());

  size_t i = 0, j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(std::move(left[i++]));
    } else {
      result.push_back(std::move(right[j++]));
    }
  }
  while (i < left.size())
    result.push_back(std::move(left[i++]));
  while (j < right.size())
    result.push_back(std::move(right[j++]));

  return result;
}

} // namespace hybrid
