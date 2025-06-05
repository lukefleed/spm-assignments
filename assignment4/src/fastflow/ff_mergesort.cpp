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
 * @brief Defines a merge or sort operation. It does not own memory, preventing
 *        allocation overhead during execution.
 */
struct MergeTask {
  Record *source;
  Record *dest;
  size_t start;
  size_t mid;
  size_t end;
};

/**
 * @class SortEmitter
 * @brief Generates tasks for the initial in-place sorting farm. Inherits from
 *        the base ff_node for source nodes.
 */
class SortEmitter : public ff_node {
public:
  SortEmitter(size_t total_size, size_t chunk_size, Record *data_ptr)
      : n(total_size), chunk_sz(chunk_size), data(data_ptr), offset(0) {}

  void *svc(void *) override {
    if (offset >= n) {
      return EOS;
    }
    size_t start = offset;
    size_t end = std::min(start + chunk_sz, n);
    offset = end;
    return new MergeTask{data, nullptr, start, 0, end};
  }

private:
  const size_t n;
  const size_t chunk_sz;
  Record *const data;
  size_t offset;
};

/**
 * @class SortWorker
 * @brief Sorts a specified region of a buffer in-place.
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::sort(task->source + task->start, task->source + task->end);
    delete task;
    return GO_ON; // No output
  }
};

/**
 * @class MergeEmitter
 * @brief Generates tasks for a parallel merge pass. Inherits from the base
 *        ff_node for source nodes.
 */
class MergeEmitter : public ff_node {
public:
  MergeEmitter(size_t total_size, size_t merge_width, Record *from_buf,
               Record *to_buf)
      : n(total_size), width(merge_width), from(from_buf), to(to_buf),
        offset(0) {}

  void *svc(void *) override {
    if (offset >= n) {
      return EOS;
    }
    size_t start = offset;
    size_t mid = std::min(start + width, n);
    size_t end = std::min(start + 2 * width, n);
    offset = end;
    return new MergeTask{from, to, start, mid, end};
  }

private:
  const size_t n;
  const size_t width;
  Record *const from;
  Record *const to;
  size_t offset;
};

/**
 * @class MergeWorker
 * @brief Merges two sorted sub-regions from a source buffer into a
 *        destination buffer, as defined by a MergeTask.
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
    return GO_ON; // No output
  }
};

} // anonymous namespace

void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1)
    return;

  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  if (n < effective_threads * 4096) {
    std::sort(data.begin(), data.end());
    return;
  }

  // --- Step 1: Initial parallel sort of chunks ---
  const size_t chunk_size =
      std::max(static_cast<size_t>(2048), n / (effective_threads * 4));

  ff_farm sort_farm;
  sort_farm.add_emitter(new SortEmitter(n, chunk_size, data.data()));
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

  // --- Step 2: Iterative parallel merge passes ---
  std::vector<Record> aux_buffer(n);
  Record *from = data.data();
  Record *to = aux_buffer.data();

  for (size_t width = chunk_size; width < n; width *= 2) {
    ff_farm merge_farm;
    merge_farm.add_emitter(new MergeEmitter(n, width, from, to));
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

  // --- Step 3: Final data placement ---
  // If 'from' points to the auxiliary buffer, the last merge pass wrote
  // its results there. We must move this final sorted data back to the
  // original 'data' vector.
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
