/**
 * @file mpi_ff_mergesort.cpp
 * @brief Hybrid MPI+FastFlow distributed mergesort implementation
 *
 * Implements scalable distributed sorting combining MPI inter-node
 * communication with FastFlow intra-node parallelization. Algorithm
 * uses binary tree reduction for hierarchical merging with payload-aware
 * buffer optimization strategies.
 *
 * DESIGN RATIONALE AND ALTERNATIVES ANALYSIS:
 *
 * Threading Model Choice (MPI_THREAD_FUNNELED):
 * - Sufficient for FastFlow's master-worker pattern where only main thread
 * communicates
 * - MPI_THREAD_MULTIPLE would add unnecessary synchronization overhead
 * communication
 * - Alternative: MPI_THREAD_SERIALIZED would be too restrictive for concurrent
 * FastFlow workers
 *
 * Communication Topology (Binary Tree vs Alternatives):
 * - Binary tree: O(log P) rounds, O(N) total data movement per process
 * - Alternative butterfly/hypercube: Same complexity but more complex
 * addressing
 * - Alternative linear reduction: O(P) rounds, unacceptable for large process
 * counts
 * - Alternative all-to-all: O(1) rounds but O(N*P) memory and communication
 * overhead. I tried to implement this, the performance was slightly worse, with
 * far more complexity in the code.
 *
 * Data Distribution Strategy (Load-Balanced Scatter):
 * - Block distribution with remainder handling ensures max difference of 1
 * element
 * - Alternative cyclic: Better load balance but destroys spatial locality
 * - Alternative master-holds-all: Creates memory bottleneck and single point of
 * failure
 * - Choice preserves cache efficiency while maintaining load balance
 *
 * Memory Management Philosophy:
 * - Move semantics throughout merge eliminates O(payload_size) copy overhead
 * - Pre-allocation strategies prevent memory fragmentation during runtime
 *
 * Key performance optimizations:
 * - Zero-payload fast path using MPI_UNSIGNED_LONG for cache efficiency
 * - Contiguous buffer packing for non-zero payloads reduces MPI overhead
 * - Move semantics throughout merge operations minimize copy costs
 * - Pre-allocation with emplace_back prevents vector reallocations
 * - Binary tree reduction ensures O(log P) communication complexity
 */

#include "mpi_ff_mergesort.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

/**
 * @brief External FastFlow parallel mergesort implementation
 *
 * Implements three-phase merge sort with farm patterns and buffer ping-ponging:
 * 1. Parallel initial sorting of cache-friendly chunks
 * 2. Iterative parallel merge passes with buffer alternation
 * 3. Final data placement ensuring in-place result semantics
 *
 * @param data Input vector sorted in-place (strong exception safety)
 * @param num_threads Worker thread count (0 defaults to single-threaded)
 */
void parallel_mergesort(std::vector<Record> &data, size_t num_threads);

namespace hybrid {

HybridMergeSort::HybridMergeSort(const HybridConfig &config)
    : config_(config), mpi_rank_(-1), mpi_size_(-1), payload_size_(0),
      metrics_{} {
  // MPI THREADING MODEL VALIDATION:
  // - MPI_THREAD_FUNNELED sufficient for FastFlow's threading model
  // - FastFlow uses master-worker pattern: only main thread communicates with
  // MPI
  // - Alternative MPI_THREAD_MULTIPLE consideration:
  //   * Pros: Maximum flexibility for multi-threaded MPI usage
  //   * Cons: Significant performance overhead from internal MPI
  //   synchronization
  int provided;
  MPI_Query_thread(&provided);
  if (provided < MPI_THREAD_FUNNELED) {
    throw std::runtime_error("MPI does not support MPI_THREAD_FUNNELED.");
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  // Note: parallel_threads must be explicitly set - no auto-detection
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
  hierarchical_merge(local_data);
  update_metrics("merge", merge_timer.elapsed_ms());

  metrics_.total_time = total_timer.elapsed_ms();
  metrics_.local_elements = (mpi_rank_ == 0) ? local_data.size() : 0;

  return local_data;
}

void HybridMergeSort::distribute_data(std::vector<Record> &local_data,
                                      const std::vector<Record> &global_data) {
  // Broadcast total size to enable consistent allocation across all processes
  size_t total_num_records = (mpi_rank_ == 0) ? global_data.size() : 0;
  MPI_Bcast(&total_num_records, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (total_num_records == 0)
    return;

  // Load-balanced distribution: remainder records distributed to first
  // processes This ensures maximum difference of 1 record between any two
  // processes
  //
  // DISTRIBUTION STRATEGY ANALYSIS:
  // - Block distribution chosen over cyclic for spatial locality preservation
  // - Remainder handling prevents load imbalance (max difference: 1 element)
  // - Alternative round-robin: destroys sequential memory access patterns
  // - Alternative random: unpredictable load balance, poor cache performance
  // - Memory-bound workloads benefit significantly from contiguous data access
  std::vector<int> send_counts(mpi_size_);
  std::vector<int> displs(mpi_size_);

  size_t base_count = total_num_records / mpi_size_;
  size_t remainder = total_num_records % mpi_size_;

  for (int i = 0; i < mpi_size_; ++i) {
    send_counts[i] = base_count + (i < static_cast<int>(remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  // Pre-allocation strategy: reserve + emplace_back prevents reallocations
  // and ensures proper Record construction with payload allocation
  local_data.clear();
  local_data.reserve(send_counts[mpi_rank_]);
  for (int i = 0; i < send_counts[mpi_rank_]; ++i) {
    local_data.emplace_back(payload_size_);
  }

  if (payload_size_ == 0) {
    // ZERO-PAYLOAD OPTIMIZATION RATIONALE:
    // - MPI_UNSIGNED_LONG provides superior cache behavior vs MPI_BYTE arrays
    // - Eliminates serialization overhead for key-only datasets
    // - Alternative: Custom MPI datatype for Record structure
    //   * Cons: MPI datatype creation overhead, padding complications
    //   * Cons: Less portable across different MPI implementations
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
    // NON-ZERO PAYLOAD COMMUNICATION STRATEGY:
    // - Contiguous buffer packing chosen over MPI derived datatypes
    // - Single MPI_Scatterv call vs multiple calls reduces latency
    // significantly
    // - Alternative: MPI_Type_create_struct for Record type
    //   * Pros: Type safety, automatic serialization
    //   * Cons: Performance penalty from MPI datatype overhead
    //   * Cons: Padding issues with different compiler/architecture
    //   combinations
    const size_t record_byte_size = sizeof(unsigned long) + payload_size_;
    std::vector<int> send_counts_bytes(mpi_size_);
    std::vector<int> displs_bytes(mpi_size_);

    for (int i = 0; i < mpi_size_; ++i) {
      send_counts_bytes[i] = send_counts[i] * record_byte_size;
      displs_bytes[i] = displs[i] * record_byte_size;
    }

    // Manual packing on root: sequential memory layout improves cache
    // performance
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

    // Direct unpacking into pre-allocated Record structures
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

  // THRESHOLD-BASED LOCAL SORTING DECISION:
  // - FastFlow overhead justification requires sufficient parallelizable work
  // - Below threshold: std::sort (introsort) provides better single-thread
  // performance
  // - Alternative: Always use parallel FastFlow
  //   * Cons: Thread creation/destruction overhead for small datasets
  //   * Cons: Farm pattern startup costs exceed sorting time for small inputs
  // - Alternative: Always use sequential std::sort
  //   * Cons: Wastes available intra-node parallelism for large local
  //   partitions
  // - Threshold tuning based on cache size and thread coordination overhead
  if (data.size() >= config_.min_local_threshold &&
      config_.parallel_threads > 1) {
    parallel_mergesort(data, config_.parallel_threads);
  } else {
    std::sort(data.begin(), data.end());
  }
}

void HybridMergeSort::hierarchical_merge(std::vector<Record> &local_data) {
  /**
   * BINARY TREE REDUCTION ALGORITHM ANALYSIS:
   *
   * Chosen topology: Binary tree reduction with O(log P) communication rounds
   *
   * Communication complexity comparison:
   * - Binary tree: O(log P) rounds, O(N) data per process, O(N*P) total network
   * traffic
   * - Linear reduction: O(P) rounds, O(N) data per process, O(N*P) total
   * network traffic
   * - Butterfly/Hypercube: O(log P) rounds, O(N*log P) data per process,
   * O(N*P*log P) total traffic
   * - All-to-all merge: O(1) rounds, O(N*P) data per process, O(N*P²) total
   * traffic
   *
   * Alternative topologies considered:
   * 1. Tournament tree (similar to binary tree but different survivor
   * selection)
   * 2. Pipeline reduction (good for streaming, poor for batch sorting)
   * 3. Mesh-based reduction (topology-aware for specific network fabrics)
   *
   * Binary tree advantages:
   * - Logarithmic depth minimizes synchronization points
   * - Each data element moves exactly log P times (optimal)
   * - Simple addressing scheme: partner = rank ± step
   * - Natural load balancing: work distributed across tree levels
   * - Memory requirements remain O(N) per process throughout
   *
   * Tree structure visualization for 8 processes:
   * Round 1: 0<--1, 2<--3, 4<--5, 6<--7  (step=1, survivors: 0,2,4,6)
   * Round 2: 0<--2, 4<--6            (step=2, survivors: 0,4)
   * Round 3: 0<--4                 (step=4, survivors: 0)
   *
   * Survivor condition: rank % (2 * step) == 0
   * Sender condition:   rank % (2 * step) == step
   * Communication partner: sender ↔ (sender - step)
   */
  for (int step = 1; step < mpi_size_; step *= 2) {
    if ((mpi_rank_ % (2 * step)) == 0) {
      // Survivor process: receive and merge data from partner
      int source = mpi_rank_ + step;
      if (source < mpi_size_) {
        // Protocol: size first, then data (enables pre-allocation)
        size_t incoming_size;
        MPI_Recv(&incoming_size, 1, MPI_UNSIGNED_LONG, source, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (incoming_size > 0) {
          // Pre-allocate partner data with proper Record construction
          std::vector<Record> partner_data;
          partner_data.reserve(incoming_size);
          for (size_t i = 0; i < incoming_size; ++i) {
            partner_data.emplace_back(payload_size_);
          }

          if (payload_size_ == 0) {
            // Zero-payload fast path: cache-efficient key-only transfer
            std::vector<unsigned long> keys(incoming_size);
            MPI_Recv(keys.data(), incoming_size, MPI_UNSIGNED_LONG, source, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t i = 0; i < incoming_size; ++i) {
              partner_data[i].key = keys[i];
            }
          } else {
            // Contiguous buffer strategy: single MPI call reduces overhead
            const size_t record_bytes = sizeof(unsigned long) + payload_size_;
            std::vector<char> buffer(incoming_size * record_bytes);

            MPI_Recv(buffer.data(), buffer.size(), MPI_BYTE, source, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Manual unpacking into pre-constructed Records
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

          // MERGE ALGORITHM SELECTION AND OPTIMIZATION:
          // - Two-way merge chosen over k-way merge for cache efficiency
          // - Move semantics optimization eliminates payload deep-copy overhead
          // - Pre-allocation prevents memory fragmentation during merge
          // operation
          if (local_data.empty()) {
            local_data = std::move(partner_data);
          } else {
            std::vector<Record> merged;
            merged.reserve(local_data.size() + partner_data.size());

            // Two-way merge using move iterators for zero-copy transfers
            // RATIONALE: Move semantics particularly critical for large
            // payloads
            // - Alternative: Copy-based merge for small payloads might be
            // faster due to
            //   reduced pointer indirection, but payload size determined at
            //   runtime
            // - Move operations amortize to O(1) for heap-allocated payloads
            // - Copy operations scale as O(payload_size) per record
            size_t i = 0, j = 0;
            while (i < local_data.size() && j < partner_data.size()) {
              if (local_data[i] < partner_data[j]) {
                merged.push_back(
                    std::move(local_data[i++])); // Zero-copy transfer
              } else {
                merged.push_back(
                    std::move(partner_data[j++])); // Zero-copy transfer
              }
            }

            // Move remaining elements from either vector
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
      // Sender process: transmit data to partner and terminate
      int target = mpi_rank_ - step;

      // Protocol: size first enables receiver pre-allocation
      size_t size = local_data.size();
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG, target, 0, MPI_COMM_WORLD);

      if (size > 0) {
        if (payload_size_ == 0) {
          // COMMUNICATION PROTOCOL DESIGN:
          // - Size-first protocol enables receiver-side pre-allocation
          // - Alternative: Self-describing messages with embedded size
          //   * Cons: Requires single large buffer allocation without size
          //   knowledge
          //   * Cons: Potential memory waste or multiple reallocation cycles
          // - Zero-payload path: Extract keys to minimize memory bandwidth
          // - Cache-conscious data layout: sequential key access patterns
          std::vector<unsigned long> keys(size);
          for (size_t i = 0; i < size; ++i) {
            keys[i] = local_data[i].key;
          }
          MPI_Send(keys.data(), size, MPI_UNSIGNED_LONG, target, 1,
                   MPI_COMM_WORLD);
        } else {
          // SENDER-SIDE PACKING STRATEGY:
          // - Manual serialization chosen over MPI derived datatypes
          // - Sequential memory layout optimizes network adapter DMA transfers
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

      // Sender terminates: data transferred to survivor in reduction tree
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
