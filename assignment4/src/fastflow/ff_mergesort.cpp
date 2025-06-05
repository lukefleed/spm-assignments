#include "../common/record.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <ff/ff.hpp>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

using namespace ff;

/**
 * @struct WorkChunk
 * @brief Lightweight work unit with direct data access
 */
struct WorkChunk {
  Record *data_start;
  size_t size;
  size_t chunk_id;

  WorkChunk(Record *start, size_t sz, size_t id)
      : data_start(start), size(sz), chunk_id(id) {}
};

/**
 * @class OptimalEmitter
 * @brief Emitter with balanced chunking strategy
 */
class OptimalEmitter : public ff_node {
private:
  Record *data_ptr;
  size_t data_size;
  size_t chunk_size;
  size_t current_offset;
  size_t total_chunks;

public:
  OptimalEmitter(std::vector<Record> &data, size_t workers)
      : data_ptr(data.data()), data_size(data.size()), current_offset(0) {

    if (data_size == 0 || workers == 0) {
      total_chunks = 0;
      return;
    }

    // Optimal chunking for maximum parallelism
    size_t target_chunks = workers * 4; // 4x over-decomposition
    chunk_size = std::max(static_cast<size_t>(1000), data_size / target_chunks);

    total_chunks = (data_size + chunk_size - 1) / chunk_size;
  }

  void *svc(void *) override {
    if (current_offset >= data_size) {
      return EOS;
    }

    size_t remaining = data_size - current_offset;
    size_t current_chunk_size = std::min(chunk_size, remaining);

    WorkChunk *chunk =
        new WorkChunk(data_ptr + current_offset, current_chunk_size,
                      current_offset / chunk_size);

    current_offset += current_chunk_size;
    return chunk;
  }

  size_t get_total_chunks() const { return total_chunks; }
};

/**
 * @class OptimalWorker
 * @brief High-performance worker with in-place sorting
 */
class OptimalWorker : public ff_node_t<WorkChunk, WorkChunk> {
public:
  WorkChunk *svc(WorkChunk *chunk) override {
    if (!chunk || chunk->size == 0) {
      return chunk;
    }

    // Use std::sort for all cases - it's highly optimized
    std::sort(chunk->data_start, chunk->data_start + chunk->size);
    return chunk;
  }
};

/**
 * @class FastMergeCollector
 * @brief High-performance collector with in-place merge
 */
class FastMergeCollector : public ff_node_t<WorkChunk, void> {
private:
  std::vector<WorkChunk *> sorted_chunks;
  std::atomic<size_t> completed_chunks{0};
  size_t expected_chunks;
  std::vector<Record> *result_ptr;

  /**
   * @brief Ultra-fast k-way merge with move semantics
   */
  void fast_k_way_merge() {
    if (sorted_chunks.empty())
      return;

    // Sort chunks by chunk_id for deterministic order
    std::sort(sorted_chunks.begin(), sorted_chunks.end(),
              [](const WorkChunk *a, const WorkChunk *b) {
                return a->chunk_id < b->chunk_id;
              });

    // Calculate total size
    size_t total_size = 0;
    for (const auto *chunk : sorted_chunks) {
      total_size += chunk->size;
    }

    // Prepare result vector
    std::vector<Record> merged_result;
    merged_result.reserve(total_size);

    if (sorted_chunks.size() == 1) {
      // Single chunk - move directly
      const auto *chunk = sorted_chunks[0];
      merged_result.reserve(chunk->size);
      for (size_t i = 0; i < chunk->size; ++i) {
        merged_result.push_back(std::move(chunk->data_start[i]));
      }
    } else {
      // Multi-way merge with priority queue
      struct HeapElement {
        Record *current;
        Record *end;

        HeapElement(Record *c, Record *e) : current(c), end(e) {}
      };

      auto comparator = [](const HeapElement &a, const HeapElement &b) {
        return a.current->key > b.current->key; // Min-heap
      };

      std::priority_queue<HeapElement, std::vector<HeapElement>,
                          decltype(comparator)>
          min_heap(comparator);

      // Initialize heap
      for (const auto *chunk : sorted_chunks) {
        if (chunk->size > 0) {
          min_heap.emplace(chunk->data_start, chunk->data_start + chunk->size);
        }
      }

      // Merge with move operations
      while (!min_heap.empty()) {
        HeapElement elem = min_heap.top();
        min_heap.pop();

        merged_result.push_back(std::move(*(elem.current)));
        ++elem.current;

        if (elem.current != elem.end) {
          min_heap.push(elem);
        }
      }
    }

    // Move result back to original vector
    *result_ptr = std::move(merged_result);

    // Cleanup
    for (auto *chunk : sorted_chunks) {
      delete chunk;
    }
    sorted_chunks.clear();
  }

public:
  explicit FastMergeCollector(std::vector<Record> *result, size_t expected)
      : expected_chunks(expected), result_ptr(result) {
    sorted_chunks.reserve(expected_chunks);
  }

  void *svc(WorkChunk *work_chunk) override {
    if (work_chunk) {
      sorted_chunks.push_back(work_chunk);

      if (++completed_chunks == expected_chunks) {
        fast_k_way_merge();
      }
    }
    return GO_ON;
  }
};

/**
 * @brief Optimized single-farm merge sort with safe operations
 * @param data_ref Reference to the vector of Records to be sorted
 * @param num_threads Total number of threads for FastFlow
 */
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data_ref,
                                     size_t num_threads) {
  if (data_ref.size() <= 1) {
    return;
  }
  if (num_threads == 0) {
    num_threads = 1;
  }

  // For small datasets, use sequential sort to avoid overhead
  if (data_ref.size() < num_threads * 5000) {
    std::sort(data_ref.begin(), data_ref.end());
    return;
  }

  // Single optimized farm
  ff_farm optimal_farm;

  // Create workers
  std::vector<ff_node *> workers;
  workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    workers.push_back(new OptimalWorker());
  }
  optimal_farm.add_workers(workers);
  optimal_farm.cleanup_workers(true);

  // Emitter
  OptimalEmitter *emitter = new OptimalEmitter(data_ref, num_threads);
  size_t expected_chunks = emitter->get_total_chunks();
  optimal_farm.add_emitter(emitter);
  optimal_farm.cleanup_emitter(true);

  // Collector
  FastMergeCollector *collector =
      new FastMergeCollector(&data_ref, expected_chunks);
  optimal_farm.add_collector(collector);
  optimal_farm.cleanup_collector(true);

  // Enable optimizations
  optimal_farm.set_scheduling_ondemand();

  // Execute farm
  if (optimal_farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Farm execution failed");
  }
}
