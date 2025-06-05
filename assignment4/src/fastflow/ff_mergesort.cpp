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
 * @struct SortChunk
 * @brief Optimized chunk with move semantics and memory safety
 */
struct SortChunk {
  std::vector<Record> data;
  size_t chunk_id;

  SortChunk() : chunk_id(0) {}
  SortChunk(std::vector<Record> d, size_t id)
      : data(std::move(d)), chunk_id(id) {}
};

/**
 * @class SafeEmitter
 * @brief Memory-safe emitter with proper chunking
 */
class SafeEmitter : public ff_node {
private:
  std::vector<std::vector<Record>> chunks;
  size_t current_chunk;

public:
  SafeEmitter(std::vector<Record> &data, size_t workers) : current_chunk(0) {
    if (data.empty() || workers == 0) {
      return;
    }

    // Optimal chunking for load balancing
    size_t total_chunks = workers * 4; // Over-decomposition
    size_t chunk_size =
        std::max(static_cast<size_t>(1),
                 (data.size() + total_chunks - 1) / total_chunks);

    // Create chunks with proper move semantics
    size_t actual_chunks = (data.size() + chunk_size - 1) / chunk_size;
    chunks.reserve(actual_chunks);

    for (size_t i = 0; i < actual_chunks; ++i) {
      size_t start = i * chunk_size;
      size_t end = std::min(start + chunk_size, data.size());

      std::vector<Record> chunk_data;
      chunk_data.reserve(end - start);

      // Move construct elements
      for (size_t j = start; j < end; ++j) {
        chunk_data.emplace_back(std::move(data[j]));
      }

      chunks.emplace_back(std::move(chunk_data));
    }

    // Clear original data
    data.clear();
  }

  void *svc(void *) override {
    if (current_chunk >= chunks.size()) {
      return EOS;
    }

    auto *chunk =
        new SortChunk(std::move(chunks[current_chunk]), current_chunk);
    ++current_chunk;
    return chunk;
  }

  size_t get_chunk_count() const { return chunks.size(); }
};

/**
 * @class OptimizedWorker
 * @brief High-performance worker with algorithm selection
 */
class OptimizedWorker : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *chunk) override {
    if (!chunk || chunk->data.empty()) {
      return chunk;
    }

    // Algorithm selection based on size
    if (chunk->data.size() <= 32) {
      // Optimized insertion sort for small arrays
      auto &data = chunk->data;
      for (size_t i = 1; i < data.size(); ++i) {
        Record key = std::move(data[i]);
        size_t j = i;
        while (j > 0 && data[j - 1].key > key.key) {
          data[j] = std::move(data[j - 1]);
          --j;
        }
        data[j] = std::move(key);
      }
    } else {
      // Standard library sort (typically introsort)
      std::sort(chunk->data.begin(), chunk->data.end());
    }

    return chunk;
  }
};

/**
 * @class OptimizedCollector
 * @brief Thread-safe collector with efficient k-way merge
 */
class OptimizedCollector : public ff_node_t<SortChunk, void> {
private:
  std::vector<Record> *result_ptr;
  std::vector<SortChunk *> chunks;
  std::atomic<size_t> received_chunks{0};
  size_t expected_chunks;

  void perform_merge() {
    if (chunks.empty())
      return;

    // Sort chunks by ID for deterministic behavior
    std::sort(chunks.begin(), chunks.end(),
              [](const SortChunk *a, const SortChunk *b) {
                return a->chunk_id < b->chunk_id;
              });

    // Calculate total size
    size_t total_size = 0;
    for (const auto *chunk : chunks) {
      total_size += chunk->data.size();
    }

    result_ptr->clear();
    result_ptr->reserve(total_size);

    if (chunks.size() == 1) {
      // Single chunk optimization
      *result_ptr = std::move(chunks[0]->data);
    } else {
      // K-way merge using priority queue with move semantics
      using Iterator = std::vector<Record>::iterator;
      using HeapElement = std::pair<Iterator, Iterator>;

      auto comp = [](const HeapElement &a, const HeapElement &b) {
        if (a.first == a.second)
          return false;
        if (b.first == b.second)
          return true;
        return a.first->key > b.first->key; // Min-heap
      };

      std::priority_queue<HeapElement, std::vector<HeapElement>, decltype(comp)>
          heap(comp);

      // Initialize heap with all chunk iterators (non-const for move)
      for (auto *chunk : chunks) {
        if (!chunk->data.empty()) {
          heap.emplace(chunk->data.begin(), chunk->data.end());
        }
      }

      // Merge all sorted chunks using move semantics
      while (!heap.empty()) {
        auto [current, end] = heap.top();
        heap.pop();

        if (current != end) {
          result_ptr->emplace_back(std::move(*current)); // Move construct
          ++current;

          if (current != end) {
            heap.emplace(current, end);
          }
        }
      }
    }

    // Cleanup
    for (auto *chunk : chunks) {
      delete chunk;
    }
    chunks.clear();
  }

public:
  explicit OptimizedCollector(std::vector<Record> *result, size_t expected)
      : result_ptr(result), expected_chunks(expected) {
    chunks.reserve(expected_chunks);
  }

  void *svc(SortChunk *chunk) override {
    if (chunk) {
      chunks.push_back(chunk);

      if (++received_chunks == expected_chunks) {
        perform_merge();
      }
    }
    return GO_ON;
  }
};

/**
 * @brief Optimized single-farm merge sort with proper memory management
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

  // For very small datasets, use sequential sort
  if (data_ref.size() < num_threads * 100) {
    std::sort(data_ref.begin(), data_ref.end());
    return;
  }

  // Create optimized farm
  ff_farm farm;

  // Create workers
  std::vector<ff_node *> workers;
  workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    workers.push_back(new OptimizedWorker());
  }
  farm.add_workers(workers);
  farm.cleanup_workers(true);

  // Create emitter and get actual chunk count
  SafeEmitter *emitter = new SafeEmitter(data_ref, num_threads);
  size_t actual_chunks = emitter->get_chunk_count();
  farm.add_emitter(emitter);
  farm.cleanup_emitter(true);

  // Create collector with correct chunk count
  OptimizedCollector *collector =
      new OptimizedCollector(&data_ref, actual_chunks);
  farm.add_collector(collector);
  farm.cleanup_collector(true);

  // Enable optimizations
  farm.set_scheduling_ondemand();

  // Execute
  if (farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Farm execution failed");
  }
}
