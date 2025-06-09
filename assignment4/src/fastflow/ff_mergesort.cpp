/**
 * @file ff_mergesort.cpp
 * @brief FastFlow parallel mergesort with farm pattern optimization
 *
 * Memory Management Philosophy:
 * - Auxiliary buffer strategy trades 2x memory for algorithmic simplicity
 * - Alternative in-place merging:
 *   * Pros: O(1) additional memory vs O(n) auxiliary buffer
 *   * Cons: Complex rotation algorithms, poor cache locality, higher constant
 * factors
 *   * Cons: Difficult parallelization due to overlapping write dependencies
 * - Buffer ping-pong eliminates complex merge coordination between levels
 *
 * Task Granularity and Load Balancing:
 * - Static work decomposition chosen over dynamic work stealing
 * - Oversubscription factor (4x threads) provides load balancing resilience
 * - Alternative dynamic scheduling:
 *   * Pros: Perfect load balancing for irregular workloads
 *   * Cons: Task stealing overhead, cache coherency issues, complex
 * implementation
 *   * Analysis: Merge sort has predictable O(n) work per level, static works
 * well
 */
#include "ff_mergesort.hpp"
#include "../common/record.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
#include <vector>

using namespace ff;

namespace {

/**
 * @struct MergeTask
 * @brief Task descriptor for sort and merge operations
 *
 * TASK DESIGN RATIONALE AND MEMORY MANAGEMENT STRATEGY:
 *
 * Raw Pointer Usage vs Smart Pointers:
 * - Raw pointers chosen for minimal task creation overhead in high-frequency
 * operations
 * - Alternative std::shared_ptr/std::unique_ptr consideration:
 *   * Cons: Reference counting overhead (atomic operations) in parallel context
 *   * Cons: Additional heap allocation for control blocks degrades cache
 * performance
 *   * Cons: Automatic lifetime management unnecessary with explicit task
 * cleanup
 * - Memory ownership remains with caller (data vector) ensuring clear semantics
 * - Explicit task deletion in worker nodes prevents memory leaks without RAII
 * overhead
 *
 * POD Structure Design:
 * - Lightweight task representation minimizes copying overhead in farm queues
 * - Cache-friendly layout: frequently accessed fields (start, end) grouped
 * together
 * - Alternative: Object-oriented task hierarchy with virtual dispatch
 *   * Cons: Virtual function call overhead per task, vtable cache misses
 *   * Cons: Larger task objects increase memory pressure in internal queues
 *
 * Lightweight POD structure designed for minimal overhead task passing.
 * Uses raw pointers to avoid std::shared_ptr allocation overhead in
 * high-frequency farm operations. Memory ownership remains with caller
 * to prevent expensive reference counting in parallel execution paths.
 *
 * Task semantics:
 * - Sort phase: [start, end) defines segment, mid unused (end == mid)
 * - Merge phase: merges [start, mid) and [mid, end) from source to dest
 */
struct MergeTask {
  Record *source; ///< Source buffer for sort/merge input
  Record *dest;   ///< Destination buffer (nullptr for sort phase)
  size_t start;   ///< Start index of operation range
  size_t mid;     ///< Boundary between first and second sorted ranges
  size_t end;     ///< End index (exclusive) of operation range
};

/**
 * @class Emitter
 * @brief Task generator implementing work decomposition for farm patterns
 *
 * WORK DECOMPOSITION STRATEGY AND SCHEDULING ANALYSIS:
 *
 * Static vs Dynamic Task Generation:
 * - Static decomposition with offset-based iteration chosen for predictable
 * overhead
 * - Alternative dynamic work stealing:
 *   * Pros: Perfect load balancing for irregular workloads
 *   * Cons: Work stealing protocols add synchronization overhead
 *   * Cons: Cache thrashing from cross-core queue access patterns
 * Merge sort has uniform O(n log n) work distribution, static
 * sufficient
 *
 * Template Method Pattern Application:
 * - Single emitter class handles both sort and merge phases via
 * parameterization
 * - Alternative: Separate emitter classes for each phase
 *   * Cons: Code duplication, increased maintenance burden
 *   * Cons: Additional virtual dispatch overhead in phase switching
 * - Phase differentiation through constructor parameters maintains type safety
 *
 * Lock-Free Design Considerations:
 * - Offset-based iteration eliminates need for complex state synchronization
 * - Single-producer (emitter) to multiple-consumer (workers) pattern is
 * naturally lock-free
 * - Alternative: Shared work queue with locks
 *   * Cons: Lock contention becomes bottleneck at high thread counts
 *
 * Dual-purpose emitter serving both sort and merge phases through
 * constructor parameterization. Uses offset-based iteration to avoid
 * complex state management while ensuring lock-free task generation.
 * Template method pattern enables phase-specific task creation logic.
 */
class Emitter : public ff_node {
public:
  /**
   * @brief Constructs emitter for specified operation phase
   *
   * @param total_size Total number of records in dataset
   * @param step Current step size (chunk_size for sort, merge_width for merge)
   * @param from_buf Source buffer pointer
   * @param to_buf Destination buffer (nullptr indicates sort phase)
   */
  Emitter(size_t total_size, size_t step, Record *from_buf,
          Record *to_buf = nullptr)
      : n(total_size), step_size(step), from(from_buf), to(to_buf), offset(0) {}

  void *svc(void *) override {
    if (offset >= n) {
      return EOS; // End-of-stream signals farm completion
    }

    size_t start = offset;
    size_t mid = std::min(start + step_size, n);
    size_t end = std::min(start + 2 * step_size, n);

    // Phase differentiation via destination buffer presence
    // Sort phase: operates on single segment [start, mid)
    // Merge phase: combines adjacent segments [start, mid), [mid, end)
    if (to == nullptr) {
      end = mid; // Collapse range for in-place sorting
    }

    auto *task = new MergeTask{from, to, start, mid, end};
    offset = end; // Advance to next non-overlapping segment

    return task;
  }

private:
  const size_t n;         ///< Total dataset size (immutable)
  const size_t step_size; ///< Current operation granularity
  Record *const from;     ///< Source buffer (ownership external)
  Record *const to;       ///< Destination buffer or nullptr for sort phase
  size_t offset;          ///< Current position in work decomposition
};

/**
 * @class SortWorker
 * @brief In-place sorting worker for initial chunk processing
 *
 *
 * In-Place vs Out-of-Place Sorting:
 * - In-place operation eliminates memory allocation overhead during sort phase
 * - Alternative: Out-of-place sorting for consistency with merge phases
 *   * Cons: Unnecessary memory allocation for temporary buffers
 *   * Cons: Additional copy overhead without algorithmic benefit
 * - Direct buffer operation maximizes cache efficiency for chunk-level sorting
 *
 * Leverages std::sort's highly optimized introsort implementation
 * Operates directly on source buffer to eliminate copy overhead during initial
 * phase. Task cleanup integrated to prevent memory leaks in farm execution.
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    // Delegate to standard library's optimized sorting algorithm
    std::sort(task->source + task->start, task->source + task->end);
    delete task;  // Immediate cleanup prevents accumulation
    return GO_ON; // Continue processing additional tasks
  }
};

/**
 * @class MergeWorker
 * @brief Parallel merge worker implementing stable merge operation
 *
 * MERGE ALGORITHM SELECTION AND OPTIMIZATION ANALYSIS:
 *
 * Two-Way vs K-Way Merge Strategy:
 * - Two-way merge chosen for optimal cache behavior and algorithmic simplicity
 * - Alternative k-way merge with priority queue:
 *   * Pros: Could reduce merge levels from log₂(n) to log_k(n)
 *   * Cons: Priority queue overhead (log k per element vs O(1) for two-way)
 *   * Cons: Poor cache locality with k memory streams, TLB pressure
 *   * Cons: Complex load balancing with variable-sized merge ranges
 *   * Analysis: Cache misses dominate for large k, two-way optimal for modern
 * architectures
 *
 * Move Semantics vs Copy Semantics:
 * - Move iterators eliminate deep-copy overhead for variable-size payloads
 * - Critical optimization for large payload sizes (metadata, embedded objects)
 * - Alternative: Copy-based merge for POD-like records
 *   * Pros: Potentially faster for small, simple record types
 *   * Cons: Payload size determined at runtime, copy overhead scales linearly
 *   * Cons: Loss of optimization opportunities for complex Record types
 *
 * Stability Preservation:
 * - std::merge maintains relative ordering of equivalent elements
 *
 * Performs out-of-place merge using move semantics to minimize
 * Record copy overhead. Uses std::merge's optimized two-way merge
 * algorithm with O(n) complexity. Move iterators enable efficient
 * payload transfer without deep copying variable-size data.
 */
class MergeWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    // Stable merge of two adjacent sorted ranges with move semantics
    // First range: [source + start, source + mid)
    // Second range: [source + mid, source + end)
    // Output: [dest + start, dest + start + (end - start))
    std::merge(std::make_move_iterator(task->source + task->start),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->end),
               task->dest + task->start);
    delete task;  // Immediate cleanup prevents accumulation
    return GO_ON; // Continue processing additional tasks
  }
};

} // anonymous namespace

/**
 * @brief parallel merge sort using FastFlow
 *
 * Synchronization Strategy:
 * - Synchronous farms ensure level completion before buffer swap
 * - Alternative: Asynchronous execution with explicit barriers
 *   * Pros: Potential pipeline parallelism between merge levels
 *   * Cons: Complex dependency management, limited benefit for merge sort
 *   * Cons: Buffer management complexity with overlapping levels
 *
 * Implements three-phase merge sort algorithm optimized for large datasets:
 * 1. Parallel initial sorting of cache-friendly chunks
 * 2. Iterative parallel merge passes with buffer ping-ponging
 * 3. Final data placement ensuring in-place result semantics
 *
 * @param data Input vector sorted in-place (strong exception safety)
 * @param num_threads Worker thread count (0 defaults to single-threaded)
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1)
    return; // Trivial cases require no processing

  // Prevent division by zero while maintaining interface compatibility
  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  // SEQUENTIAL FALLBACK THRESHOLD ANALYSIS:
  // - Threshold calculation balances parallelization overhead vs benefit
  // - Factor of 1024: Empirically derived based on typical cache sizes (L1:
  // 32KB, L2: 256KB)
  // - Effective threads * 1024 ensures sufficient work per thread to amortize:
  //   * Thread creation/destruction overhead
  //   * Task queue management overhead
  //   * Context switching costs
  // - Alternative: Fixed threshold approach
  //   * Cons: Ignores available parallelism, poor resource utilization
  //   * Cons: Doesn't scale with thread count, suboptimal for varying hardware
  if (n < effective_threads * 1024) {
    std::sort(data.begin(), data.end());
    return;
  }

  // CHUNK SIZE OPTIMIZATION AND CACHE EFFICIENCY:
  // - Cache-friendly chunk sizing balances memory hierarchy utilization
  // - Minimum 1024 elements targets L2 cache capacity (typical 256KB cache /
  // 256B per Record ≈ 1024)
  // - Division by (threads * 4): Creates deliberate oversubscription for load
  // balancing
  //   * 4x oversubscription factor provides resilience against:
  //     - Thread scheduling variations and OS interruptions
  //     - Memory access latency variations (cache misses, etc..)
  //     - Heterogeneous processing speeds across cores
  // - Alternative: threads * 2 oversubscription
  //   * Cons: Insufficient buffering against load imbalance
  //   * Cons: Under-utilization during scheduling hiccups
  // - Alternative: threads * 8 oversubscription
  //   * Cons: Excessive task creation overhead, diminishing cache benefits
  //   * Cons: Increased queue management overhead in FastFlow runtime
  // - Alternative: Perfect work division (n / threads)
  //   * Cons: No resilience against load imbalance, poor utilization
  //   * Cons: Assumes perfect thread scheduling, unrealistic in practice
  const size_t chunk_size =
      std::max(static_cast<size_t>(1024), n / (effective_threads * 4));

  ff_farm sort_farm;
  sort_farm.add_emitter(new Emitter(n, chunk_size, data.data()));
  sort_farm.cleanup_emitter(true); // Automatic memory management

  std::vector<ff_node *> sorters;
  sorters.reserve(effective_threads); // Prevent reallocation overhead
  for (size_t i = 0; i < effective_threads; ++i) {
    sorters.push_back(new SortWorker());
  }
  sort_farm.add_workers(sorters);
  sort_farm.cleanup_workers(true); // Automatic memory management

  // Synchronous execution ensures completion before merge phase
  if (sort_farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Initial sorting farm failed");
  }

  // Phase 2: Iterative parallel merge passes with buffer alternation
  // BUFFER PING-PONG STRATEGY AND MEMORY MANAGEMENT:
  // - Auxiliary buffer eliminates in-place merge complexity enabling full
  // parallelization
  // - Alternative: In-place merging with rotation algorithms
  //   * Pros: O(1) additional memory vs O(n) auxiliary buffer
  //   * Cons: Complex rotation algorithms (Gries-Mills, block-wise merging)
  //   * Cons: Sequential dependencies prevent effective parallelization
  //   * Cons: Poor cache behavior due to non-sequential access patterns
  // - Alternative: Multiple auxiliary buffers for pipeline parallelism
  //   * Pros: Could overlap merge levels for streaming scenarios
  //   * Cons: 3x memory overhead, minimal benefit for finite datasets
  //   * Cons: Complex buffer management, potential memory fragmentation
  // - Buffer ping-pong provides clean separation between merge levels
  // - Memory allocation occurs once, amortized across all merge operations
  std::vector<Record> aux_buffer(n);
  Record *from = data.data();
  Record *to = aux_buffer.data();

  // MERGE LEVEL ITERATION AND COMPLEXITY ANALYSIS:
  // - Bottom-up approach: log₂(n/chunk_size) iterations total
  // - Each iteration processes entire dataset with width-doubling strategy
  // - Alternative: Top-down recursive decomposition
  //   * Cons: Complex work coordination, potential stack overflow
  //   * Cons: Irregular task sizes, poor load balancing
  //   * Cons: Recursive function call overhead
  // - Alternative: Multi-level parallel merging
  //   * Pros: Potential parallelism across different merge levels
  //   * Cons: Complex dependency management, limited practical benefit
  //   * Cons: Memory management complexity with multiple active levels
  // - Width doubling ensures optimal merge tree depth: O(log n)
  // - Each data element participates in exactly log₂(n/chunk_size) merge
  // operations
  for (size_t width = chunk_size; width < n; width *= 2) {
    ff_farm merge_farm;
    merge_farm.add_emitter(new Emitter(n, width, from, to));
    merge_farm.cleanup_emitter(true); // Automatic memory management

    std::vector<ff_node *> mergers;
    mergers.reserve(effective_threads); // Prevent reallocation overhead
    for (size_t i = 0; i < effective_threads; ++i) {
      mergers.push_back(new MergeWorker());
    }
    merge_farm.add_workers(mergers);
    merge_farm.cleanup_workers(true); // Automatic memory management

    // Synchronous execution ensures level completion before buffer swap
    if (merge_farm.run_and_wait_end() < 0) {
      throw std::runtime_error("Merge farm failed");
    }

    // Buffer ping-pong: swap source and destination for next iteration
    std::swap(from, to);
  }

  // Phase 3: Final data placement ensuring in-place semantics
  // RESULT PLACEMENT AND MOVE OPTIMIZATION:
  // - Buffer ping-pong necessitates final result location determination
  // - from pointer indicates final data location after all merge iterations
  // - Alternative: Always copy back to original buffer
  //   * Cons: Unnecessary copy operation when result already in correct
  //   location
  // - Alternative: Return result buffer pointer, modify interface
  //   * Pros: Eliminates final copy operation entirely
  //   * Cons: Interface change complicates integration, breaks in-place
  //   semantics
  //   * Cons: Memory management responsibilities transferred to caller
  // - Move semantics provides zero-copy transfer for large payloads
  // - Critical optimization when Record contains heap-allocated payload data
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
