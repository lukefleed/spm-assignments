#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator> // For std::make_move_iterator
#include <omp.h>
#include <stdexcept>
#include <vector>

// Forward declaration for the FastFlow-based local sorter.
// This is assumed to be defined elsewhere and is used for intra-node sorting.
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace {

/**
 * @brief Packs a vector of Record objects into a byte buffer for MPI
 * communication.
 * @param records The records to pack.
 * @param buffer The output byte buffer.
 * @param payload_size The size of the payload for each record.
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
 * @brief Unpacks a byte buffer into a vector of Record objects.
 * @param buffer The input byte buffer.
 * @param num_records The number of records to unpack from the buffer.
 * @param records The output vector of records.
 * @param payload_size The size of the payload for each record.
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

  // Phase 1: Distribute data from rank 0 to all processes.
  Timer dist_timer;
  distribute_data(local_data, data);
  update_metrics("distribution", dist_timer.elapsed_ms());

  // Phase 2: Sort local data chunks in parallel.
  Timer sort_timer;
  sort_local_data(local_data);
  update_metrics("local_sort", sort_timer.elapsed_ms());

  // Phase 3: Merge sorted chunks in a balanced, recursive manner.
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

  // Calculate counts and displacements for MPI_Scatterv.
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);
  size_t base_count = total_num_records / mpi_size_;
  size_t remainder = total_num_records % mpi_size_;
  for (int i = 0; i < mpi_size_; ++i) {
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // Convert record counts to byte counts for MPI.
  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  std::vector<int> send_counts_bytes(mpi_size_);
  std::vector<int> displs_bytes(mpi_size_);
  for (int i = 0; i < mpi_size_; ++i) {
    send_counts_bytes[i] = send_counts[i] * record_byte_size;
    displs_bytes[i] = displs[i] * record_byte_size;
  }

  // Rank 0 packs all data into a single send buffer.
  std::vector<char> send_buffer;
  if (mpi_rank_ == 0) {
    pack_records(global_data, send_buffer, payload_size_);
  }

  // Scatter data to all processes.
  std::vector<char> recv_buffer(send_counts_bytes[mpi_rank_]);
  MPI_Scatterv(send_buffer.data(), send_counts_bytes.data(),
               displs_bytes.data(), MPI_BYTE, recv_buffer.data(),
               recv_buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Unpack received bytes into Record objects.
  unpack_records(recv_buffer.data(), send_counts[mpi_rank_], local_data,
                 payload_size_);
  metrics_.bytes_communicated += recv_buffer.size();
}

void HybridMergeSort::sort_local_data(std::vector<Record> &data) {
  if (data.empty())
    return;

  // Use a parallel sort (e.g., FastFlow-based) for large local data chunks.
  if (data.size() >= config_.min_local_threshold &&
      config_.parallel_threads > 1) {
    parallel_mergesort(data, config_.parallel_threads);
  } else {
    // Fallback to sequential sort for smaller chunks.
    std::sort(data.begin(), data.end());
  }
}

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;

  // This loop implements the recursive doubling (binary-tree) merge pattern.
  // In each step, processes are paired up, one sends its data, the other
  // receives and merges. The distance between partners (`step`) doubles in each
  // iteration.
  for (int step = 1; step < mpi_size_; step *= 2) {
    // Determine the role of the current process in this step.
    if (mpi_rank_ % (2 * step) != 0) {
      // Role: Sender. This process sends its data to its partner and becomes
      // inactive.
      int partner_rank = mpi_rank_ - step;
      std::vector<char> send_buffer;
      pack_records(local_data, send_buffer, payload_size_);
      MPI_Send(send_buffer.data(), send_buffer.size(), MPI_BYTE, partner_rank,
               0, MPI_COMM_WORLD);

      // Once data is sent, this process has no more work to do.
      local_data.clear();
      break;
    }

    // Role: Receiver. This process will receive data from its partner if the
    // partner exists.
    int partner_rank = mpi_rank_ + step;
    if (partner_rank < mpi_size_) {
      // Probe to get the size of the incoming message.
      MPI_Status status;
      MPI_Probe(partner_rank, 0, MPI_COMM_WORLD, &status);
      int incoming_bytes;
      MPI_Get_count(&status, MPI_BYTE, &incoming_bytes);

      if (incoming_bytes > 0) {
        // Receive the data.
        std::vector<char> recv_buffer(incoming_bytes);
        MPI_Recv(recv_buffer.data(), incoming_bytes, MPI_BYTE, partner_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        metrics_.bytes_communicated += incoming_bytes;

        // Unpack the data into a temporary vector.
        std::vector<Record> partner_data;
        unpack_records(recv_buffer.data(), incoming_bytes / record_byte_size,
                       partner_data, payload_size_);

        // Merge the received data with the local data.
        parallel_merge(local_data, partner_data);
      }
    }
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

  // Use sequential merge for small total sizes to avoid parallel overhead.
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

  // Parallel merge implementation using OpenMP.
  std::vector<Record> merged(total_size);

  // Ensure A is the larger vector for better pivot selection.
  std::vector<Record> &A =
      (local_data.size() >= partner_data.size()) ? local_data : partner_data;
  std::vector<Record> &B =
      (local_data.size() >= partner_data.size()) ? partner_data : local_data;

  const int num_threads = config_.parallel_threads;
  std::vector<size_t> split_A(num_threads + 1, 0);
  std::vector<size_t> split_B(num_threads + 1, 0);

// Phase 1: Partition the larger array A into `num_threads` chunks.
// For each pivot in A, find its corresponding split point in B.
#pragma omp parallel for
  for (int i = 1; i < num_threads; ++i) {
    split_A[i] = (A.size() * i) / num_threads;
    auto it = std::lower_bound(B.begin(), B.end(), A[split_A[i]]);
    split_B[i] = std::distance(B.begin(), it);
  }
  split_A[num_threads] = A.size();
  split_B[num_threads] = B.size();

// Phase 2: Each thread merges its assigned sub-arrays into the final
// destination.
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
