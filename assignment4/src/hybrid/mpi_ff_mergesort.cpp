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
 * @brief Packs a vector of Record objects into a contiguous byte buffer for
 * serialization.
 *
 * This function serializes Record objects by copying their key and payload data
 * into a linear byte buffer. Each record is packed with the key
 * followed by the payload data. The buffer is resized to accommodate all
 * records.
 *
 * @param records The vector of Record objects to be packed
 * @param buffer The output buffer that will contain the serialized data
 * (resized automatically)
 * @param payload_size The size in bytes of each record's payload data
 *
 * @note The buffer layout for each record is: [key (8 bytes)][payload
 * (payload_size bytes)]
 * @note If payload_size is 0 or rec.payload is null, only the key is copied but
 * space is still reserved
 */
void pack_records(const std::vector<Record> &records, std::vector<char> &buffer,
                  size_t payload_size) {
  const size_t record_byte_size = sizeof(unsigned long) + payload_size;
  buffer.resize(records.size() * record_byte_size);
  char *base_ptr = buffer.data();

#pragma omp parallel for
  for (size_t i = 0; i < records.size(); ++i) {
    // use const reference to avoid copying
    const auto &rec = records[i];
    // Calculate the pointer to the start of the current record in the buffer
    char *ptr = base_ptr + i * record_byte_size;
    // Copy the key into the buffer
    memcpy(ptr, &rec.key, sizeof(unsigned long));
    // Move the pointer forward by the size of the key
    ptr += sizeof(unsigned long);
    // If payload_size > 0 and rec.payload is not null, copy the payload data
    if (payload_size > 0 && rec.payload) {
      // Copy the payload data into the buffer
      memcpy(ptr, rec.payload, payload_size);
    }
  }
}

/**
 * @brief Unpacks serialized records from a buffer into a vector of Record
 * objects.
 *
 * This function deserializes a contiguous buffer containing packed Record data
 * back into a vector of Record objects. Each record in the buffer consists of
 * a key (unsigned long) followed by optional payload data.
 *
 * @param buffer Pointer to the serialized data buffer containing packed records
 * @param num_records Number of records to unpack from the buffer
 * @param records Reference to vector that will be populated with unpacked
 * Record objects
 * @param payload_size Size in bytes of the payload data for each record (0 if
 * no payload)
 *
 * @note The records vector is cleared and resized to accommodate the unpacked
 * data. Memory layout in buffer: [key1][payload1][key2][payload2]...
 * @warning No bounds checking is performed on the buffer - caller must ensure
 *          buffer contains at least num_records * (sizeof(unsigned long) +
 * payload_size) bytes
 */
void unpack_records(const char *buffer, size_t num_records,
                    std::vector<Record> &records, size_t payload_size) {
  records.clear();              // Clear existing records
  records.reserve(num_records); // Reserve space for new records

  // Pre-allocate all records first
  for (size_t i = 0; i < num_records; ++i) {
    records.emplace_back(payload_size);
  }

  // Calculate the size of each record in bytes
  // Each record consists of a key (unsigned long) + payload data
  const size_t record_byte_size = sizeof(unsigned long) + payload_size;

#pragma omp parallel for
  for (size_t i = 0; i < num_records; ++i) {
    auto &rec = records[i];
    // Calculate the pointer to the start of the current record in the buffer
    const char *ptr = buffer + i * record_byte_size;
    // Copy the key into the record
    memcpy(&rec.key, ptr, sizeof(unsigned long));
    // Move the pointer forward by the size of the key
    ptr += sizeof(unsigned long);
    // If payload_size > 0 and rec.payload is not null, copy the payload data
    if (payload_size > 0 && rec.payload) {
      // Copy the payload data into the record
      memcpy(rec.payload, ptr, payload_size);
    }
  }
}

} // namespace

namespace hybrid {

/**
 * @brief Constructs a HybridMergeSort object with the specified configuration.
 *
 * Initializes the hybrid merge sort implementation that combines MPI and
 * FastFlow. Performs validation checks to ensure MPI is properly initialized
 * and supports the required threading level (MPI_THREAD_FUNNELED). Sets up MPI
 * communicator information including rank and size.
 *
 * @param config The hybrid configuration containing parallel threading settings
 *               and other algorithm parameters
 *
 * @throws std::runtime_error If MPI is not initialized or does not support
 *                           MPI_THREAD_FUNNELED threading level
 * @throws std::invalid_argument If parallel_threads in config is set to 0
 */
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

/**
 * @brief Performs hybrid merge sort on a distributed dataset using MPI and
 * FastFlow.
 *
 * This method implements a three-phase distributed sorting algorithm:
 * 1. Distribution phase: Distributes input data from rank 0 to all MPI
 * processes
 * 2. Local sorting phase: Each process sorts its local data chunk in parallel
 * using FastFlow
 * 3. Merge phase: Hierarchically merges sorted chunks across processes using
 * MPI communication
 *
 * @param data Input vector of Record objects to be sorted (only meaningful on
 * rank 0)
 * @param payload_size Size of the payload data for each record
 * @return std::vector<Record> Sorted vector of records (populated only on rank
 * 0, empty on other ranks)
 *
 * @note The input data parameter is only used by rank 0; other processes
 * receive their data chunks through MPI communication during the distribution
 * phase
 */
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

/**
 * @brief Distributes data from rank 0 to all MPI processes for parallel
 * processing.
 *
 * This function implements a data distribution strategy where the root process
 * (rank 0) broadcasts the total number of records to all processes, then
 * scatters the data evenly across all MPI ranks using MPI_Scatterv. The
 * distribution handles uneven data sizes by giving remainder records to
 * lower-ranked processes.
 *
 * @param local_data [out] Vector to store the records assigned to this MPI
 * process
 * @param global_data [in] Complete dataset available only on rank 0; empty on
 * other ranks
 *
 * @details The function performs the following steps:
 * 1. Broadcasts total record count from rank 0 to all processes
 * 2. Calculates how many records each process should receive (load balancing)
 * 3. Converts record counts to byte counts for MPI communication
 * 4. Packs records into a contiguous buffer on rank 0
 * 5. Scatters data chunks to all processes using MPI_Scatterv
 * 6. Unpacks received bytes back into Record objects on each process
 * 7. Updates communication metrics with bytes transferred
 *
 * @note Records are distributed as evenly as possible, with any remainder
 * records assigned to the lowest-ranked processes (0, 1, 2, ...).
 * @note The function handles empty datasets by early return.
 */
void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  size_t total_num_records = (mpi_rank_ == 0) ? global_data.size() : 0;
  // Broadcast total record count to all processes
  MPI_Bcast(&total_num_records, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (total_num_records == 0)
    return;

  // Calculate counts and displacements for MPI_Scatterv.
  // Ensure that the difference of records between two processes is at most 1.
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);
  size_t base_count = total_num_records / mpi_size_;
  size_t remainder =
      total_num_records %
      mpi_size_; // Number of extra records when not evenly divisible
  for (int i = 0; i < mpi_size_; ++i) {
    // Give one extra record to the first 'remainder' processes (ranks 0, 1, 2,
    // ...) This ensures even distribution when total_num_records is not
    // divisible by mpi_size_
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // Convert record counts to byte counts for MPI.
  // MPI_Scatterv requires byte-level communication, so we need to convert
  // the number of records each process should receive into the corresponding
  // number of bytes that need to be transmitted.
  const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
  std::vector<int> send_counts_bytes(mpi_size_);
  std::vector<int> displs_bytes(mpi_size_);
  for (int i = 0; i < mpi_size_; ++i) {
    // Calculate how many bytes to send to process i
    // Each record consists of a key (unsigned long) + payload data
    send_counts_bytes[i] = send_counts[i] * record_byte_size;

    // Calculate byte offset where process i's data starts in the send buffer
    // This tells MPI_Scatterv where to find each process's data chunk
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

/**
 * @brief Sorts local data using either parallel or sequential merge sort based
 * on data size and configuration.
 *
 * This method intelligently chooses between parallel and sequential sorting
 * algorithms depending on the size of the input data and the configured
 * threading parameters. For large datasets that meet the minimum threshold and
 * when multiple threads are available, it uses a parallel merge sort
 * implementation. For smaller datasets or single-threaded configurations, it
 * falls back to the standard library's sequential sort algorithm.
 *
 * @param data Reference to a vector of Record objects to be sorted in-place.
 *             The vector is modified directly and will be sorted upon
 * completion. If the vector is empty, the function returns immediately without
 *             performing any operations.
 */
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

/**
 * @brief Performs hierarchical merge of sorted data across MPI processes using
 * recursive doubling pattern.
 *
 * This function implements a binary-tree merge pattern where processes are
 * paired up in each iteration to merge their sorted data. The distance between
 * partner processes doubles in each step (1, 2, 4, 8). In each iteration, one
 * process acts as a sender (transmits its data and becomes inactive) while the
 * other acts as a receiver (receives and merges the data).
 *
 * The algorithm follows these steps:
 * 1. Processes with rank % (2 * step) != 0 send their data to partner (rank -
 * step) and exit
 * 2. Processes with rank % (2 * step) == 0 receive data from partner (rank +
 * step) if it exists
 * 3. Receiving processes merge the incoming data with their local data using
 * parallel merge
 * 4. The step size doubles and the process repeats until only one process
 * remains with all data
 *
 * @param local_data Reference to vector containing the locally sorted records
 * that will be merged with data from other processes. After completion, only
 * the root process (rank 0) will contain the fully merged and sorted data.
 *
 * @note This function modifies the local_data vector in place. Sender processes
 * will have their local_data cleared after transmission.
 * @note Assumes that local_data is already sorted before calling this function.
 */
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
      MPI_Status status; // Status object to hold probe results
      MPI_Probe(partner_rank, 0, MPI_COMM_WORLD,
                &status); // MPI_Probe is blocking: the process waits until a
                          // message is available from partner_rank starts
                          // arriving. This does not transfer any data, but
                          // populates the status object with information about
                          // the incoming message (sender, tag, size).
      int incoming_bytes;
      MPI_Get_count(
          &status, MPI_BYTE,
          &incoming_bytes); // Get the size of the incoming message in bytes.

      if (incoming_bytes > 0) {
        // Just after that MPI_Probe returns the exact size of the incoming
        // message, we can allocate a buffer to hold the data
        std::vector<char> recv_buffer(
            incoming_bytes); // Allocate buffer to hold incoming data.
        MPI_Recv(
            recv_buffer.data(), incoming_bytes, MPI_BYTE, partner_rank, 0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE); // Then MPI_Recv is called to actually receive
                                // the data from the partner process. This is a
                                // blocking call: the process waits until all
                                // incoming_bytes have been received and stored
                                // in recv_buffer.

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

/**
 * @brief Merges two sorted vectors in parallel using OpenMP threading.
 *
 * This function merges the local_data vector with the partner_data vector,
 * storing the result in local_data. The merge operation uses parallel
 * processing for large datasets to improve performance, falling back to
 * sequential merge for smaller datasets to avoid parallel overhead.
 *
 * Algorithm:
 * 1. For small datasets (< 20000 elements) or single-threaded configuration,
 *    performs sequential merge using std::merge
 * 2. For larger datasets, uses a two-phase parallel merge:
 *    - Phase 1: Partitions the larger array into num_threads chunks and finds
 *      corresponding split points in the smaller array using binary search
 *    - Phase 2: Each thread independently merges its assigned sub-arrays
 *      into the final result vector
 *
 * @param local_data The first sorted vector to merge (modified in-place with
 * result)
 * @param partner_data The second sorted vector to merge (contents moved during
 * merge)
 *
 * @post partner_data is left in a valid but unspecified state due to move
 * operations
 */
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
    // Reuse merge buffer instead of allocating new memory
    if (merge_buffer_.size() < total_size) {
      merge_buffer_.resize(total_size);
    }

    std::merge(std::make_move_iterator(local_data.begin()),
               std::make_move_iterator(local_data.end()),
               std::make_move_iterator(partner_data.begin()),
               std::make_move_iterator(partner_data.end()),
               merge_buffer_.begin());

    // Swap buffers instead of expensive move assignment
    local_data.swap(merge_buffer_);
    merge_buffer_.resize(total_size); // Keep capacity for next use
    return;
  }

  // Parallel merge implementation using OpenMP with double buffering
  if (merge_buffer_.size() < total_size) {
    merge_buffer_.resize(total_size);
  }

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

// Phase 2: Each thread merges its assigned sub-arrays into the merge buffer.
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
               merge_buffer_.begin() + output_start);
  }

  // Swap buffers instead of expensive move assignment
  local_data.swap(merge_buffer_);
  merge_buffer_.resize(total_size); // Keep capacity for next use
}

// Metrics update function to record elapsed time for different phases.
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
