#ifndef MPI_COMMUNICATION_HPP
#define MPI_COMMUNICATION_HPP

#include "../common/record.hpp"
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @brief High-performance MPI communication utilities for Record structures
 *
 * Optimized for minimal overhead and maximum throughput in distributed
 * mergesort
 */
namespace mpi_comm {

/**
 * @brief MPI datatype wrapper for Record structures
 *
 * Efficiently handles variable-payload Record transmission without
 * serialization overhead
 */
class RecordDatatype {
private:
  MPI_Datatype record_type_;
  size_t payload_size_;
  bool initialized_;

public:
  explicit RecordDatatype(size_t payload_size);
  ~RecordDatatype();

  RecordDatatype(const RecordDatatype &) = delete;
  RecordDatatype &operator=(const RecordDatatype &) = delete;

  MPI_Datatype get() const { return record_type_; }
  size_t payload_size() const { return payload_size_; }
  bool is_initialized() const { return initialized_; }
};

/**
 * @brief Non-blocking communication manager for optimal
 * computation-communication overlap
 *
 * Handles asynchronous Record transfers with automatic buffer management
 */
class AsyncCommManager {
private:
  struct PendingTransfer {
    MPI_Request request;
    std::vector<Record> buffer;
    std::vector<char> raw_buffer; // Add this for persistent buffer
    int partner_rank;
    bool is_send;

    PendingTransfer(size_t buffer_size, int rank, bool send)
        : partner_rank(rank), is_send(send) {
      buffer.reserve(buffer_size);
      request = MPI_REQUEST_NULL;
    }
  };

  std::vector<std::unique_ptr<PendingTransfer>> pending_transfers_;
  const RecordDatatype &datatype_;

public:
  explicit AsyncCommManager(const RecordDatatype &dt) : datatype_(dt) {}
  ~AsyncCommManager();

  /**
   * @brief Initiates non-blocking send operation
   * @param data Source data to send
   * @param dest_rank Target MPI rank
   * @param tag Message tag
   * @return Transfer ID for completion checking
   */
  size_t async_send(const std::vector<Record> &data, int dest_rank, int tag);

  /**
   * @brief Initiates non-blocking receive operation
   * @param buffer_size Expected number of records
   * @param src_rank Source MPI rank
   * @param tag Message tag
   * @return Transfer ID for completion checking
   */
  size_t async_recv(size_t buffer_size, int src_rank, int tag);

  /**
   * @brief Checks if transfer is complete
   * @param transfer_id ID returned by async_send/recv
   * @return True if complete, false if still pending
   */
  bool is_complete(size_t transfer_id);

  /**
   * @brief Waits for transfer completion and retrieves data
   * @param transfer_id ID returned by async_send/recv
   * @return Received data (empty for send operations)
   */
  std::vector<Record> wait_and_get(size_t transfer_id);

  /**
   * @brief Waits for all pending transfers to complete
   */
  void wait_all();

  /**
   * @brief Returns number of pending transfers
   */
  size_t pending_count() const { return pending_transfers_.size(); }
};

/**
 * @brief Optimized data distribution for load balancing
 *
 * Calculates optimal chunk sizes considering payload overhead and network
 * topology
 */
class DataDistributor {
private:
  int rank_;
  int size_;
  size_t total_elements_;
  size_t payload_size_;

public:
  DataDistributor(int rank, int size, size_t total_elements,
                  size_t payload_size)
      : rank_(rank), size_(size), total_elements_(total_elements),
        payload_size_(payload_size) {}

  /**
   * @brief Calculates local partition size for current rank
   */
  size_t local_size() const;

  /**
   * @brief Calculates starting index for current rank's partition
   */
  size_t local_start() const;

  /**
   * @brief Calculates partition size for specific rank
   */
  size_t size_for_rank(int target_rank) const;

  /**
   * @brief Calculates starting index for specific rank
   */
  size_t start_for_rank(int target_rank) const;

  /**
   * @brief Determines if data distribution is balanced
   */
  bool is_balanced() const;
};

/**
 * @brief Hierarchical merge tree communication pattern
 *
 * Implements optimal merge tree topology for minimizing communication rounds
 */
class MergeTree {
private:
  int rank_;
  int size_;
  int height_;

public:
  MergeTree(int rank, int size);

  /**
   * @brief Determines if current rank participates in given level
   */
  bool is_active_at_level(int level) const;

  /**
   * @brief Returns partner rank for merging at given level
   */
  int get_partner(int level) const;

  /**
   * @brief Determines if current rank is receiver at given level
   */
  bool is_receiver(int level) const;

  /**
   * @brief Returns tree height (number of merge levels)
   */
  int height() const { return height_; }

  /**
   * @brief Returns final result holder rank
   */
  int root_rank() const { return 0; }
};

/**
 * @brief Memory-efficient all-to-all data redistribution
 *
 * Optimized for minimal memory usage during data redistribution phases
 */
void efficient_alltoallv(const std::vector<Record> &send_data,
                         const std::vector<int> &send_counts,
                         const std::vector<int> &send_displs,
                         std::vector<Record> &recv_data,
                         const std::vector<int> &recv_counts,
                         const std::vector<int> &recv_displs,
                         const RecordDatatype &datatype, MPI_Comm comm);

/**
 * @brief Optimized collective operations for Record vectors
 */
namespace collective {

/**
 * @brief Broadcasts Record vector from root to all processes
 */
void bcast_records(std::vector<Record> &data, int root,
                   const RecordDatatype &datatype, MPI_Comm comm);

/**
 * @brief Gathers Record vectors from all processes to root
 */
void gather_records(const std::vector<Record> &send_data,
                    std::vector<Record> &recv_data,
                    const std::vector<int> &recv_counts, int root,
                    const RecordDatatype &datatype, MPI_Comm comm);

/**
 * @brief Scatters Record vector from root to all processes
 */
void scatter_records(const std::vector<Record> &send_data,
                     std::vector<Record> &recv_data,
                     const std::vector<int> &send_counts, int root,
                     const RecordDatatype &datatype, MPI_Comm comm);
} // namespace collective

} // namespace mpi_comm

#endif // MPI_COMMUNICATION_HPP
