#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include <algorithm>
#include <cmath> // For std::min with size_t and std::sqrt
#include <cstring>
#include <omp.h> // Include OpenMP header
#include <stdexcept>
#include <vector>

// Forward declaration for the FastFlow-based local sorter.
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace {

/**
 * @brief Serializes a vector of Records into a flat byte buffer for MPI
 * transfer.
 */
void pack_records(const std::vector<Record> &records, std::vector<char> &buffer,
                  size_t payload_size) {
  const size_t record_byte_size = sizeof(unsigned long) + payload_size;
  buffer.resize(records.size() * record_byte_size);
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

/**
 * @brief Deserializes a flat byte buffer into a vector of Records.
 */
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

HybridMergeSort::HybridMergeSort(const HybridConfig &config)
    : config_(config), mpi_rank_(-1), mpi_size_(-1), payload_size_(0),
      metrics_{} {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("MPI must be initialized");
  }
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error("MPI does not support MPI_THREAD_FUNNELED");
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
  if (config_.parallel_threads == 0) {
    throw std::invalid_argument("parallel_threads must be explicitly set");
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
    hierarchical_merge(local_data);
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

  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  std::vector<int> send_counts_bytes(mpi_size_);
  std::vector<int> displs_bytes(mpi_size_);
  for (int i = 0; i < mpi_size_; ++i) {
    send_counts_bytes[i] = send_counts[i] * record_byte_size;
    displs_bytes[i] = displs[i] * record_byte_size;
  }

  std::vector<char> send_buffer;
  if (mpi_rank_ == 0) {
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

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  int k =
      (mpi_size_ >= 4) ? 4 : 2; // k-nomial factor. Use k=4 for 4+ processes.
  k = std::min(k, mpi_size_);
  if (k <= 1)
    return;

  // --- Phase 1: Merge within sub-groups ---
  int color = mpi_rank_ / k;
  int group_rank = mpi_rank_ % k;
  MPI_Comm group_comm;
  // This is a collective call, all processes in MPI_COMM_WORLD must call it.
  MPI_Comm_split(MPI_COMM_WORLD, color, group_rank, &group_comm);

  int group_size;
  MPI_Comm_size(group_comm, &group_size);

  if (group_rank != 0) {
    // Senders send their data to the group leader (rank 0 in group_comm).
    // Using a blocking send is safe here as the receiver posts non-blocking
    // receives.
    std::vector<char> send_buffer;
    pack_records(local_data, send_buffer, payload_size_);
    MPI_Send(send_buffer.data(), send_buffer.size(), MPI_BYTE, 0, 0,
             group_comm);
  } else {
    // Group leader receives data from all other members non-blockingly to
    // prevent deadlock.
    std::vector<std::vector<char>> recv_buffers(group_size);
    std::vector<MPI_Request> requests;
    for (int i = 1; i < group_size; ++i) {
      MPI_Status status;
      MPI_Probe(i, 0, group_comm, &status); // Find message size first.
      int incoming_bytes;
      MPI_Get_count(&status, MPI_BYTE, &incoming_bytes);
      if (incoming_bytes > 0) {
        recv_buffers[i].resize(incoming_bytes);
        MPI_Request req;
        MPI_Irecv(recv_buffers[i].data(), incoming_bytes, MPI_BYTE, i, 0,
                  group_comm, &req);
        requests.push_back(req);
      }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // After all data is received, merge it.
    for (int i = 1; i < group_size; ++i) {
      if (!recv_buffers[i].empty()) {
        std::vector<Record> partner_data;
        const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
        unpack_records(recv_buffers[i].data(),
                       recv_buffers[i].size() / record_byte_size, partner_data,
                       payload_size_);
        parallel_merge(local_data, partner_data);
      }
    }
  }
  MPI_Comm_free(&group_comm);

  // Processes that were senders are now idle and have no more data.
  // They must still participate in the next collective call to avoid deadlock.
  if (group_rank != 0) {
    local_data.clear();
  }

  // --- Phase 2: Merge between group leaders ---
  // All processes MUST call MPI_Comm_split. Non-leaders use MPI_UNDEFINED
  // to signal they should not be part of the new communicator.
  int leader_color = (mpi_rank_ % k == 0) ? 0 : MPI_UNDEFINED;
  MPI_Comm leader_comm;
  MPI_Comm_split(MPI_COMM_WORLD, leader_color, mpi_rank_, &leader_comm);

  if (leader_comm != MPI_COMM_NULL) {
    int leader_rank, leader_size;
    MPI_Comm_rank(leader_comm, &leader_rank);
    MPI_Comm_size(leader_comm, &leader_size);

    // A standard, robust binary merge among the small number of leaders.
    for (int step = 1; step < leader_size; step *= 2) {
      if (leader_rank % (2 * step) != 0) {
        int target_rank = leader_rank - step;
        std::vector<char> send_buffer;
        pack_records(local_data, send_buffer, payload_size_);
        MPI_Send(send_buffer.data(), send_buffer.size(), MPI_BYTE, target_rank,
                 0, leader_comm);
        local_data.clear();
        break;
      }
      int source_rank = leader_rank + step;
      if (source_rank < leader_size) {
        MPI_Status status;
        MPI_Probe(source_rank, 0, leader_comm, &status);
        int incoming_bytes;
        MPI_Get_count(&status, MPI_BYTE, &incoming_bytes);
        if (incoming_bytes > 0) {
          std::vector<char> recv_buffer(incoming_bytes);
          MPI_Recv(recv_buffer.data(), incoming_bytes, MPI_BYTE, source_rank, 0,
                   leader_comm, MPI_STATUS_IGNORE);
          std::vector<Record> partner_data;
          const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
          unpack_records(recv_buffer.data(),
                         recv_buffer.size() / record_byte_size, partner_data,
                         payload_size_);
          parallel_merge(local_data, partner_data);
        }
      }
    }
    MPI_Comm_free(&leader_comm);
  }
}

void HybridMergeSort::parallel_merge(std::vector<Record> &local_data,
                                     std::vector<Record> &partner_data) {
  if (partner_data.empty())
    return;
  if (local_data.empty()) {
    local_data = std::move(partner_data);
    return;
  }
  const size_t total_size = local_data.size() + partner_data.size();
  const size_t parallel_threshold = 20000;

  if (total_size < parallel_threshold || config_.parallel_threads <= 1) {
    std::vector<Record> merged;
    merged.reserve(total_size);
    std::merge(std::make_move_iterator(local_data.begin()),
               std::make_move_iterator(local_data.end()),
               std::make_move_iterator(partner_data.begin()),
               std::make_move_iterator(partner_data.end()),
               std::back_inserter(merged));
    local_data = std::move(merged);
    return;
  }

  std::vector<Record> merged(total_size);
  std::vector<Record> &A =
      (local_data.size() >= partner_data.size()) ? local_data : partner_data;
  std::vector<Record> &B =
      (local_data.size() >= partner_data.size()) ? partner_data : local_data;

  const int num_threads = config_.parallel_threads;
  std::vector<size_t> split_A(num_threads + 1, 0);
  std::vector<size_t> split_B(num_threads + 1, 0);

#pragma omp parallel for
  for (int i = 1; i < num_threads; ++i) {
    split_A[i] = (A.size() * i) / num_threads;
    auto it = std::lower_bound(B.begin(), B.end(), A[split_A[i]]);
    split_B[i] = std::distance(B.begin(), it);
  }
  split_A[num_threads] = A.size();
  split_B[num_threads] = B.size();

#pragma omp parallel for
  for (int i = 0; i < num_threads; ++i) {
    size_t start_A = split_A[i];
    size_t end_A = split_A[i + 1];
    size_t start_B = split_B[i];
    size_t end_B = split_B[i + 1];
    size_t output_start = start_A + start_B;

    std::merge(std::make_move_iterator(A.begin() + start_A),
               std::make_move_iterator(A.begin() + end_A),
               std::make_move_iterator(B.begin() + start_B),
               std::make_move_iterator(B.begin() + end_B),
               merged.begin() + output_start);
  }
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
