#include "mpi_communication.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace mpi_comm {

// ============================================================================
// RecordDatatype Implementation
// ============================================================================

RecordDatatype::RecordDatatype(size_t payload_size)
    : payload_size_(payload_size), initialized_(false) {

  if (payload_size == 0) {
    MPI_Type_contiguous(sizeof(unsigned long), MPI_BYTE, &record_type_);
  } else {
    int block_lengths[2] = {static_cast<int>(sizeof(unsigned long)),
                            static_cast<int>(payload_size)};
    MPI_Aint displacements[2] = {0,
                                 static_cast<MPI_Aint>(sizeof(unsigned long))};
    MPI_Datatype types[2] = {MPI_BYTE, MPI_BYTE};

    MPI_Type_create_struct(2, block_lengths, displacements, types,
                           &record_type_);
  }

  MPI_Type_commit(&record_type_);
  initialized_ = true;
}

RecordDatatype::~RecordDatatype() {
  if (initialized_) {
    MPI_Type_free(&record_type_);
  }
}

// ============================================================================
// AsyncCommManager Implementation
// ============================================================================

AsyncCommManager::~AsyncCommManager() { wait_all(); }

size_t AsyncCommManager::async_send(const std::vector<Record> &data,
                                    int dest_rank, int tag) {
  auto transfer =
      std::make_unique<PendingTransfer>(data.size(), dest_rank, true);

  // Allocate persistent send buffer
  size_t record_size = sizeof(unsigned long) + datatype_.payload_size();
  transfer->raw_buffer.resize(data.size() * record_size);
  char *buf_ptr = transfer->raw_buffer.data();

  for (const auto &record : data) {
    std::memcpy(buf_ptr, &record.key, sizeof(unsigned long));
    buf_ptr += sizeof(unsigned long);

    if (datatype_.payload_size() > 0 && record.payload) {
      std::memcpy(buf_ptr, record.payload, datatype_.payload_size());
    }
    buf_ptr += datatype_.payload_size();
  }

  MPI_Isend(transfer->raw_buffer.data(), data.size(), datatype_.get(),
            dest_rank, tag, MPI_COMM_WORLD, &transfer->request);

  size_t transfer_id = pending_transfers_.size();
  pending_transfers_.push_back(std::move(transfer));

  return transfer_id;
}

size_t AsyncCommManager::async_recv(size_t buffer_size, int src_rank, int tag) {
  auto transfer =
      std::make_unique<PendingTransfer>(buffer_size, src_rank, false);
  transfer->buffer.resize(buffer_size);

  // Allocate payload for received records
  for (auto &record : transfer->buffer) {
    if (datatype_.payload_size() > 0) {
      record.payload = new char[datatype_.payload_size()];
      record.payload_size = datatype_.payload_size();
    }
  }

  // Allocate persistent receive buffer
  size_t record_size = sizeof(unsigned long) + datatype_.payload_size();
  transfer->raw_buffer.resize(buffer_size * record_size);

  MPI_Irecv(transfer->raw_buffer.data(), buffer_size, datatype_.get(), src_rank,
            tag, MPI_COMM_WORLD, &transfer->request);

  size_t transfer_id = pending_transfers_.size();
  pending_transfers_.push_back(std::move(transfer));

  return transfer_id;
}

bool AsyncCommManager::is_complete(size_t transfer_id) {
  if (transfer_id >= pending_transfers_.size())
    return false;

  int flag;
  MPI_Test(&pending_transfers_[transfer_id]->request, &flag, MPI_STATUS_IGNORE);
  return flag != 0;
}

std::vector<Record> AsyncCommManager::wait_and_get(size_t transfer_id) {
  if (transfer_id >= pending_transfers_.size()) {
    return {};
  }

  auto &transfer = pending_transfers_[transfer_id];
  MPI_Wait(&transfer->request, MPI_STATUS_IGNORE);

  std::vector<Record> result;
  if (!transfer->is_send) {
    // Unpack received data from raw buffer
    char *buf_ptr = transfer->raw_buffer.data();
    size_t record_size = sizeof(unsigned long) + datatype_.payload_size();

    for (auto &record : transfer->buffer) {
      std::memcpy(&record.key, buf_ptr, sizeof(unsigned long));
      buf_ptr += sizeof(unsigned long);

      if (datatype_.payload_size() > 0) {
        std::memcpy(record.payload, buf_ptr, datatype_.payload_size());
      }
      buf_ptr += datatype_.payload_size();
    }

    result = std::move(transfer->buffer);
  }

  // Clean up this transfer
  pending_transfers_[transfer_id].reset();
  return result;
}

void AsyncCommManager::wait_all() {
  for (auto &transfer : pending_transfers_) {
    if (transfer && transfer->request != MPI_REQUEST_NULL) {
      MPI_Wait(&transfer->request, MPI_STATUS_IGNORE);
    }
  }
  pending_transfers_.clear();
}

// ============================================================================
// Collective Operations
// ============================================================================

void efficient_alltoallv(const std::vector<Record> &send_data,
                         const std::vector<int> &send_counts,
                         const std::vector<int> &send_displs,
                         std::vector<Record> &recv_data,
                         const std::vector<int> &recv_counts,
                         const std::vector<int> &recv_displs,
                         const RecordDatatype &datatype, MPI_Comm comm) {

  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // Calculate total send and receive sizes
  size_t total_send = 0, total_recv = 0;
  for (int i = 0; i < comm_size; ++i) {
    total_send += send_counts[i];
    total_recv += recv_counts[i];
  }

  // Create contiguous send buffer
  size_t record_size = sizeof(unsigned long) + datatype.payload_size();
  std::vector<char> send_buffer(total_send * record_size);
  std::vector<char> recv_buffer(total_recv * record_size);

  // Pack send data according to send_displs
  for (int proc = 0; proc < comm_size; ++proc) {
    if (send_counts[proc] > 0) {
      char *buf_ptr = send_buffer.data() + send_displs[proc] * record_size;

      for (int i = 0; i < send_counts[proc]; ++i) {
        const auto &record = send_data[send_displs[proc] + i];
        std::memcpy(buf_ptr, &record.key, sizeof(unsigned long));
        buf_ptr += sizeof(unsigned long);

        if (datatype.payload_size() > 0 && record.payload) {
          std::memcpy(buf_ptr, record.payload, datatype.payload_size());
        }
        buf_ptr += datatype.payload_size();
      }
    }
  }

  // Perform actual MPI_Alltoallv
  MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                datatype.get(), recv_buffer.data(), recv_counts.data(),
                recv_displs.data(), datatype.get(), comm);

  // Unpack received data
  recv_data.resize(total_recv);
  char *buf_ptr = recv_buffer.data();

  for (size_t i = 0; i < total_recv; ++i) {
    std::memcpy(&recv_data[i].key, buf_ptr, sizeof(unsigned long));
    buf_ptr += sizeof(unsigned long);

    if (datatype.payload_size() > 0) {
      recv_data[i].payload = new char[datatype.payload_size()];
      recv_data[i].payload_size = datatype.payload_size();
      std::memcpy(recv_data[i].payload, buf_ptr, datatype.payload_size());
    }
    buf_ptr += datatype.payload_size();
  }
}

namespace collective {

void scatter_records(const std::vector<Record> &send_data,
                     std::vector<Record> &recv_data,
                     const std::vector<int> &send_counts, int root,
                     const RecordDatatype &datatype, MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  // First, broadcast send_counts to all processes
  std::vector<int> counts_copy(send_counts);
  if (comm_rank != root) {
    counts_copy.resize(comm_size);
  }
  MPI_Bcast(counts_copy.data(), comm_size, MPI_INT, root, comm);

  // Get receive count for this rank
  int recv_count = counts_copy[comm_rank];

  std::vector<char> send_buffer;
  std::vector<int> displs;

  if (comm_rank == root) {
    // Create send buffer and displacement array
    size_t record_size = sizeof(unsigned long) + datatype.payload_size();
    send_buffer.resize(send_data.size() * record_size);
    displs.resize(comm_size);
    displs[0] = 0;

    for (int i = 1; i < comm_size; ++i) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }

    // Pack send data
    char *buf_ptr = send_buffer.data();
    for (const auto &record : send_data) {
      std::memcpy(buf_ptr, &record.key, sizeof(unsigned long));
      buf_ptr += sizeof(unsigned long);

      if (datatype.payload_size() > 0 && record.payload) {
        std::memcpy(buf_ptr, record.payload, datatype.payload_size());
      }
      buf_ptr += datatype.payload_size();
    }
  }

  // Prepare receive buffer
  recv_data.resize(recv_count);
  for (auto &record : recv_data) {
    if (datatype.payload_size() > 0) {
      record.payload = new char[datatype.payload_size()];
      record.payload_size = datatype.payload_size();
    }
  }

  size_t record_size = sizeof(unsigned long) + datatype.payload_size();
  std::vector<char> recv_buffer(recv_count * record_size);

  MPI_Scatterv(send_buffer.data(), counts_copy.data(), displs.data(),
               datatype.get(), recv_buffer.data(), recv_count, datatype.get(),
               root, comm);

  // Unpack received data
  char *buf_ptr = recv_buffer.data();
  for (auto &record : recv_data) {
    std::memcpy(&record.key, buf_ptr, sizeof(unsigned long));
    buf_ptr += sizeof(unsigned long);

    if (datatype.payload_size() > 0) {
      std::memcpy(record.payload, buf_ptr, datatype.payload_size());
    }
    buf_ptr += datatype.payload_size();
  }
}

} // namespace collective

// ============================================================================
// DataDistributor Implementation
// ============================================================================

size_t DataDistributor::local_size() const { return size_for_rank(rank_); }

size_t DataDistributor::local_start() const { return start_for_rank(rank_); }

size_t DataDistributor::size_for_rank(int target_rank) const {
  if (target_rank < 0 || target_rank >= size_) {
    return 0;
  }

  size_t base_size = total_elements_ / size_;
  size_t remainder = total_elements_ % size_;

  // Distribute remainder among first 'remainder' ranks
  return base_size + (target_rank < static_cast<int>(remainder) ? 1 : 0);
}

size_t DataDistributor::start_for_rank(int target_rank) const {
  if (target_rank < 0 || target_rank >= size_) {
    return total_elements_;
  }

  size_t base_size = total_elements_ / size_;
  size_t remainder = total_elements_ % size_;

  // Calculate start position considering remainder distribution
  size_t start = target_rank * base_size;
  if (target_rank < static_cast<int>(remainder)) {
    start += target_rank;
  } else {
    start += remainder;
  }

  return start;
}

bool DataDistributor::is_balanced() const {
  if (size_ <= 1)
    return true;

  size_t min_size = size_for_rank(size_ - 1);
  size_t max_size = size_for_rank(0);

  // Consider balanced if difference is at most 1
  return (max_size - min_size) <= 1;
}

// ============================================================================
// MergeTree Implementation
// ============================================================================

MergeTree::MergeTree(int rank, int size) : rank_(rank), size_(size) {
  // Calculate tree height (log2(size), rounded up)
  height_ = 0;
  int temp_size = size;
  while (temp_size > 1) {
    temp_size = (temp_size + 1) / 2;
    height_++;
  }
}

bool MergeTree::is_active_at_level(int level) const {
  if (level < 0 || level >= height_) {
    return false;
  }

  // At level l, processes participate if they are divisible by 2^l
  int step = 1 << level;

  // Check if this rank participates at this level
  if ((rank_ % step) != 0) {
    return false;
  }

  // Calculate potential partner
  int receiver_step = 1 << (level + 1);
  bool is_potential_receiver = (rank_ % receiver_step) == 0;

  if (is_potential_receiver) {
    // This is a potential receiver, check if sender partner exists
    int partner = rank_ + step;
    return partner < size_;
  } else {
    // This is a potential sender, check if receiver partner exists
    int partner = rank_ - step;
    return partner >= 0;
  }
}

int MergeTree::get_partner(int level) const {
  if (!is_active_at_level(level)) {
    return -1;
  }

  int step = 1 << level;
  int receiver_step = 1 << (level + 1);

  if ((rank_ % receiver_step) == 0) {
    // This is a receiver, partner is rank + step
    int partner = rank_ + step;
    return (partner < size_) ? partner : -1;
  } else {
    // This is a sender, partner is rank - step
    int partner = rank_ - step;
    return (partner >= 0) ? partner : -1;
  }
}

bool MergeTree::is_receiver(int level) const {
  if (!is_active_at_level(level)) {
    return false;
  }

  int receiver_step = 1 << (level + 1);
  return (rank_ % receiver_step) == 0;
}

} // namespace mpi_comm
