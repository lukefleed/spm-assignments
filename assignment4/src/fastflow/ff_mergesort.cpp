#include "ff_mergesort.hpp"
#include "../common/record.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
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
 * @brief Task generator for farm work decomposition
 *
 * Dual-purpose emitter serving both sort and merge phases.
 * Uses offset-based iteration for lock-free task generation.
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

  void *svc(void *) override {
    if (offset >= n) {
      return EOS; // Signal farm completion
    }

    size_t start = offset;
    size_t mid = std::min(start + step_size, n);
    size_t end = std::min(start + 2 * step_size, n);

    // Sort phase: operates on single segment [start, mid)
    // Merge phase: combines adjacent segments [start, mid), [mid, end)
    if (to == nullptr) {
      end = mid; // Collapse range for in-place sorting
    }

    auto *task = new MergeTask{from, to, start, mid, end};
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
 * @brief In-place sorting worker for initial chunk processing
 *
 * Operates directly on source buffer to eliminate copy overhead.
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::sort(task->source + task->start, task->source + task->end);
    delete task; // Immediate cleanup
    return GO_ON;
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
 * @brief Parallel merge sort using FastFlow framework
 *
 * Three-phase algorithm:
 * 1. Parallel sorting of cache-friendly chunks
 * 2. Iterative parallel merge passes with buffer ping-ponging
 * 3. Final data placement ensuring in-place result
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

  // Cache-friendly chunk sizing with 4x oversubscription for load balancing
  const size_t chunk_size =
      std::max(static_cast<size_t>(1024), n / (effective_threads * 4));

  // Phase 1: Parallel initial sorting
  ff_farm sort_farm;
  sort_farm.add_emitter(new Emitter(n, chunk_size, data.data()));
  sort_farm.cleanup_emitter(true);

  std::vector<ff_node *> sorters;
  sorters.reserve(effective_threads);
  for (size_t i = 0; i < effective_threads; ++i) {
    sorters.push_back(new SortWorker());
  }
  sort_farm.add_workers(sorters);
  sort_farm.cleanup_workers(true);

  if (sort_farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Initial sorting farm failed");
  }

  // Phase 2: Iterative parallel merge with buffer ping-pong
  std::vector<Record> aux_buffer(n);
  Record *from = data.data();
  Record *to = aux_buffer.data();

  // Bottom-up merge with width doubling
  for (size_t width = chunk_size; width < n; width *= 2) {
    ff_farm merge_farm;
    merge_farm.add_emitter(new Emitter(n, width, from, to));
    merge_farm.cleanup_emitter(true);

    std::vector<ff_node *> mergers;
    mergers.reserve(effective_threads);
    for (size_t i = 0; i < effective_threads; ++i) {
      mergers.push_back(new MergeWorker());
    }
    merge_farm.add_workers(mergers);
    merge_farm.cleanup_workers(true);

    if (merge_farm.run_and_wait_end() < 0) {
      throw std::runtime_error("Merge farm failed");
    }

    // Swap buffers for next iteration
    std::swap(from, to);
  }

  // Phase 3: Final data placement if needed
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
