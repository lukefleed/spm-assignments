#include "ff_mergesort.hpp"
#include "../common/record.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <vector>

using namespace ff;

namespace {

/**
 * @brief Task descriptor for sort and merge operations
 *
 * Uses raw pointers to minimize task overhead in high-frequency operations.
 * Memory ownership remains with caller to avoid reference counting overhead.
 */
struct MergeTask {
  Record *source; ///< Source buffer for operation
  Record *dest;   ///< Destination buffer (nullptr for sort phase)
  size_t start;   ///< Start index of operation range
  size_t mid;     ///< Boundary between sorted ranges
  size_t end;     ///< End index (exclusive) of operation range
};

/**
 * @brief Emitter node for FastFlow-based merge sort implementation
 *
 * The Emitter class extends ff_node to distribute work tasks across worker
 * nodes in a farm pattern. It operates in two distinct phases:
 *
 * 1. **Sort Phase**: Distributes individual chunks for parallel sorting
 *    - Uses only the source buffer (to_buf is nullptr)
 *    - Creates tasks for segments [start, mid) of size step_size
 *    - Performs in-place sorting operations
 *
 * 2. **Merge Phase**: Distributes merge operations for sorted segments
 *    - Uses both source and destination buffers
 *    - Creates tasks to merge adjacent sorted segments [start, mid) and [mid,
 * end)
 *    - step_size represents the width of segments to be merged
 *
 * The emitter maintains internal state to track progress through the dataset
 * and automatically signals completion when all segments have been distributed.
 *
 * @note The emitter determines operation mode based on whether to_buf is
 * nullptr
 */
class Emitter : public ff_node {
public:
  /**
   * @brief Initialize emitter for specified operation phase
   * @param total_size Total number of records
   * @param step Current step size (chunk_size for sort, merge_width for merge)
   * @param from_buf Source buffer pointer
   * @param to_buf Destination buffer (nullptr for sort phase)
   */
  Emitter(size_t total_size, size_t step, Record *from_buf,
          Record *to_buf = nullptr)
      : n(total_size), step_size(step), from(from_buf), to(to_buf), offset(0) {}

  // The offset variable works as a cursor to track the portion of the array
  // already processed. It is initialized to 0 and incremented by step_size
  // after each task creation. When it reaches or exceeds n, the emitter signals
  // EOS to indicate that all tasks have been created and no more work is
  // available.
  void *svc(void *) override {
    if (offset >= n) {
      return EOS; // Signal farm completion
    }

    size_t start = offset;
    size_t mid = std::min(start + step_size, n);
    size_t end = std::min(start + 2 * step_size, n); // Range for the merge

    // Sort phase: operates on single segment [start, mid)
    // Merge phase: combines adjacent segments [start, mid), [mid, end)
    if (to == nullptr) { // Then we are in the sort phase
      end = mid;         // Collapse range for in-place sorting
    }

    auto *task = new MergeTask{from, to, start, mid, end}; // Create a new task
    offset = end; // Advance to next segment

    return task;
  }

private:
  const size_t n;         ///< Total dataset size
  const size_t step_size; ///< Current operation granularity
  Record *const from;     ///< Source buffer
  Record *const to;       ///< Destination buffer or nullptr
  size_t offset;          ///< Current position in decomposition
};

/**
 * @brief Worker node for parallel sorting operations in FastFlow framework.
 *
 * SortWorker is a FastFlow node that processes MergeTask objects by sorting
 * specified ranges of data using std::sort. This class inherits from ff_node_t.
 *
 * The worker takes ownership of the MergeTask memory and is responsible for
 * cleaning it up after processing. Each worker operates independently on
 * different data ranges.
 *
 * @tparam Input type: MergeTask* - Task containing source array and range
 * information
 * @tparam Output type: void - No output is produced, results are stored
 * in-place
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::sort(task->source + task->start, task->source + task->end);
    delete task;  // Immediate cleanup, the worker is the owner of the memory of
                  // the task, it takes the responsibility to delete it.
    return GO_ON; // Signal that the worker is ready for the next task
  }
};

/**
 * @brief Parallel merge worker with move semantics optimization
 *
 * Uses move iterators to minimize Record copy overhead for variable-size
 * payloads.
 */
class MergeWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    // Stable merge of two adjacent sorted ranges
    // Using std::make_move_iterator is crucial. When we pass to std::merge two
    // move iterators, it does not copy the elements, but moves them (it does
    // not use `operator=` but `operator=(&&)`). This is important for
    // our payload, as it avoids unnecessary copies. During the
    // merge, instead of allocating a new buffer, we steal from the record the
    // source and assign it to the destination. Then the source pointer is
    // nulled out.
    std::merge(std::make_move_iterator(task->source + task->start),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->end),
               task->dest + task->start);
    delete task; // Immediate cleanup
    return GO_ON;
  }
};

} // anonymous namespace

/**
 * @brief Performs parallel merge sort on a vector of Record objects using
 * FastFlow framework.
 *
 * This function implements a three-phase parallel merge sort algorithm:
 * 1. Initial parallel sorting of chunks using a farm pattern
 * 2. Iterative bottom-up merging with buffer ping-pong technique
 * 3. Final data placement to ensure results are in the original vector
 *
 * The algorithm uses oversubscription (4x the number of threads) for better
 * load balancing and includes optimizations such as sequential fallback for
 * small datasets and efficient memory management with buffer swapping.
 *
 * @param data The vector of Record objects to be sorted in-place
 * @param num_threads The number of worker threads to use for parallel
 * execution. If 0, defaults to 1 thread.
 *
 * @throws std::runtime_error If any of the FastFlow farms fail to execute
 * properly
 *
 * @note For datasets smaller than num_threads * 1024 elements, the function
 * falls back to sequential std::sort to avoid parallelization overhead.
 * @note The minimum chunk size is 1024 elements to ensure efficient parallel
 * processing.
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1)
    return;

  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  // Sequential fallback for small datasets to avoid parallelization overhead
  if (n < effective_threads * 1024) {
    std::sort(data.begin(), data.end());
    return;
  }

  // Chunk sizing with 4x oversubscription for load balancing
  const size_t chunk_size =
      std::max(static_cast<size_t>(1024), n / (effective_threads * 4));

  // Phase 1: Parallel initial sorting
  ff_farm sort_farm;
  sort_farm.add_emitter(new Emitter(n, chunk_size, data.data()));
  sort_farm.cleanup_emitter(true);

  std::vector<ff_node *> sorters;     // Workers for sorting phase
  sorters.reserve(effective_threads); // Preallocate vector
  for (size_t i = 0; i < effective_threads; ++i) {
    // Create a worker for each thread
    sorters.push_back(new SortWorker());
  }
  // Add all workers to the farm
  sort_farm.add_workers(sorters);
  sort_farm.cleanup_workers(true); // Cleanup workers after use

  // Run the sorting farm and wait for completion
  if (sort_farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Initial sorting farm failed");
  }

  // Phase 2: Iterative parallel merge with buffer ping-pong
  std::vector<Record> aux_buffer(n); // Single allocation at the beginning
  Record *from = data.data();        // Pointer to current source buffer
  Record *to = aux_buffer.data();    // Pointer to current destination buffer

  // Bottom-up merge with width doubling
  // The size of the segments doubles each iteration
  for (size_t width = chunk_size; width < n; width *= 2) {
    ff_farm merge_farm;
    // The emitter gets configured to read from `from` and write to `to`
    merge_farm.add_emitter(new Emitter(n, width, from, to));
    merge_farm.cleanup_emitter(true);

    std::vector<ff_node *> mergers;     // Workers for merging phase
    mergers.reserve(effective_threads); // Preallocate vector
    for (size_t i = 0; i < effective_threads; ++i) {
      // Create a worker for each thread
      mergers.push_back(new MergeWorker());
    }

    // Add all workers to the farm
    merge_farm.add_workers(mergers);
    merge_farm.cleanup_workers(true); // Cleanup workers after use

    // Run the merge farm and wait for completion
    if (merge_farm.run_and_wait_end() < 0) {
      throw std::runtime_error("Merge farm failed");
    }

    // Swap buffers for next iteration
    std::swap(from, to);
  }

  // Phase 3: Final data placement if needed
  // The data may still be in the auxiliary buffer
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
