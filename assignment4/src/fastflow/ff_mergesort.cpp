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
 * Lightweight task representation that does not own memory,
 * preventing allocation overhead during execution.
 */
struct MergeTask {
  Record *source;
  Record *dest;
  size_t start;
  size_t mid;
  size_t end;
};

/**
 * @class Emitter
 * @brief Task generator for sort and merge phases
 */
class Emitter : public ff_node {
public:
  Emitter(size_t total_size, size_t step, Record *from_buf,
          Record *to_buf = nullptr)
      : n(total_size), step_size(step), from(from_buf), to(to_buf), offset(0) {}

  void *svc(void *) override {
    if (offset >= n) {
      return EOS;
    }

    size_t start = offset;
    size_t mid = std::min(start + step_size, n);
    size_t end = std::min(start + 2 * step_size, n);

    // Initial sort phase: task defines a segment to be sorted in-place,
    // identified by start and end (which equals mid).
    if (to == nullptr) {
      end = mid;
    }

    auto *task = new MergeTask{from, to, start, mid, end};
    offset = end;

    return task;
  }

private:
  const size_t n;
  const size_t step_size;
  Record *const from;
  Record *const to;
  size_t offset;
};

/**
 * @class SortWorker
 * @brief Sorts a specified region of a buffer in-place
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::sort(task->source + task->start, task->source + task->end);
    delete task;
    return GO_ON;
  }
};

/**
 * @class MergeWorker
 * @brief Merges two sorted sub-regions from source to destination buffer
 */
class MergeWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::merge(std::make_move_iterator(task->source + task->start),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->end),
               task->dest + task->start);
    delete task;
    return GO_ON;
  }
};

} // anonymous namespace

/**
 * @brief Parallel merge sort implementation using FastFlow framework
 *
 * Implements a multi-stage parallel merge sort algorithm with synchronized
 * farm patterns. The algorithm divides the input into fixed-size chunks for
 * initial sorting, followed by iterative parallel merge phases until the
 * entire dataset is sorted.
 *
 * @param data Input vector of Record objects to sort in-place
 * @param num_threads Number of worker threads for parallel execution
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1)
    return;

  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  if (n < effective_threads * 1024) {
    std::sort(data.begin(), data.end());
    return;
  }

  // Phase 1: Parallel in-place sorting of initial chunks
  const size_t chunk_size =
      std::max(static_cast<size_t>(1024), n / (effective_threads * 4));

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

  // Phase 2: Synchronized parallel merge passes
  std::vector<Record> aux_buffer(n);
  Record *from = data.data();
  Record *to = aux_buffer.data();

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

    std::swap(from, to);
  }

  // Phase 3: Final data movement to original buffer
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
