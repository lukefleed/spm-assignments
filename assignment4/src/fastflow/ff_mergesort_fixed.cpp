#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include <algorithm>
#include <deque>
#include <ff/ff.hpp>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

using namespace ff;

/**
 * @struct SortChunk
 * @brief Represents a chunk of data with optimized memory management
 */
struct SortChunk {
  std::vector<Record> data;
  size_t chunk_id;

  SortChunk() : chunk_id(0) {}
  SortChunk(std::vector<Record> d, size_t id)
      : data(std::move(d)), chunk_id(id) {}
};

/**
 * @class OptimizedSortEmitter
 * @brief High-performance emitter that pre-chunks data for optimal load
 * balancing
 */
class OptimizedSortEmitter : public ff_node {
private:
  std::vector<std::vector<Record>> prepared_chunks;
  size_t current_chunk_idx;
  size_t total_chunks;

public:
  OptimizedSortEmitter(std::vector<Record> &original_data, size_t num_workers)
      : current_chunk_idx(0), total_chunks(0) {

    if (original_data.empty()) {
      return;
    }

    // Over-decompose for better load balancing - more chunks than workers
    size_t optimal_chunks = std::max(num_workers * 4, static_cast<size_t>(1));
    size_t chunk_size =
        (original_data.size() + optimal_chunks - 1) / optimal_chunks;
    chunk_size = std::max(static_cast<size_t>(1), chunk_size);

    total_chunks = (original_data.size() + chunk_size - 1) / chunk_size;
    prepared_chunks.reserve(total_chunks);

    // Pre-chunk the data for optimal performance
    for (size_t i = 0; i < total_chunks; ++i) {
      size_t start = i * chunk_size;
      size_t end = std::min(start + chunk_size, original_data.size());

      std::vector<Record> chunk_data;
      chunk_data.reserve(end - start);

      // Move data efficiently
      for (size_t j = start; j < end; ++j) {
        chunk_data.emplace_back(std::move(original_data[j]));
      }

      prepared_chunks.emplace_back(std::move(chunk_data));
    }

    // Clear original data to free memory
    original_data.clear();
  }

  void *svc(void * /*task*/) override {
    if (current_chunk_idx >= total_chunks) {
      return EOS;
    }

    auto *chunk = new SortChunk(std::move(prepared_chunks[current_chunk_idx]),
                                current_chunk_idx);
    current_chunk_idx++;
    return chunk;
  }

  size_t get_total_chunks() const { return total_chunks; }
};

/**
 * @class HighPerformanceSortWorker
 * @brief Optimized worker for sorting with minimal overhead
 */
class HighPerformanceSortWorker : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *chunk) override {
    if (chunk == (SortChunk *)EOS) {
      return (SortChunk *)EOS;
    }

    // Use optimized sort for performance
    std::sort(chunk->data.begin(), chunk->data.end());
    return chunk;
  }
};

/**
 * @class ForwardingCollector
 * @brief Minimal collector for maximum throughput
 */
class ForwardingCollector : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *task) override { return task; }
};

/**
 * @class OptimizedFinalMerge
 * @brief High-performance final merge using priority queue for k-way merge
 */
class OptimizedFinalMerge : public ff_node_t<SortChunk, void> {
private:
  std::vector<Record> *final_result_ptr;
  std::deque<SortChunk *> chunks_buffer;

  void optimized_k_way_merge() {
    if (!final_result_ptr || chunks_buffer.empty()) {
      return;
    }

    // Handle single chunk case efficiently
    if (chunks_buffer.size() == 1) {
      *final_result_ptr = std::move(chunks_buffer.front()->data);
      delete chunks_buffer.front();
      chunks_buffer.clear();
      return;
    }

    // Calculate total size for optimal memory allocation
    size_t total_size = 0;
    for (const auto *chunk : chunks_buffer) {
      if (chunk) {
        total_size += chunk->data.size();
      }
    }

    final_result_ptr->clear();
    final_result_ptr->reserve(total_size);

    // Use priority queue for efficient k-way merge
    using RecordIterator = std::vector<Record>::iterator;
    using IteratorPair = std::pair<RecordIterator, RecordIterator>;

    auto comparator = [](const IteratorPair &a, const IteratorPair &b) {
      if (a.first == a.second)
        return false;
      if (b.first == b.second)
        return true;
      return a.first->key > b.first->key; // Min-heap based on key
    };

    std::priority_queue<IteratorPair, std::vector<IteratorPair>,
                        decltype(comparator)>
        min_heap(comparator);

    // Initialize heap with iterators from all chunks
    for (auto *chunk : chunks_buffer) {
      if (chunk && !chunk->data.empty()) {
        min_heap.push({chunk->data.begin(), chunk->data.end()});
      }
    }

    // Perform k-way merge
    while (!min_heap.empty()) {
      auto [current_it, end_it] = min_heap.top();
      min_heap.pop();

      if (current_it != end_it) {
        final_result_ptr->push_back(std::move(*current_it));
        ++current_it;

        if (current_it != end_it) {
          min_heap.push({current_it, end_it});
        }
      }
    }

    // Clean up chunks
    for (auto *chunk : chunks_buffer) {
      delete chunk;
    }
    chunks_buffer.clear();
  }

public:
  explicit OptimizedFinalMerge(std::vector<Record> *result_ptr)
      : final_result_ptr(result_ptr) {}

  ~OptimizedFinalMerge() {
    for (auto *chunk : chunks_buffer) {
      delete chunk;
    }
  }

  void *svc(SortChunk *chunk) override {
    if (chunk) {
      chunks_buffer.push_back(chunk);
    }
    return GO_ON;
  }

  void svc_end() override { optimized_k_way_merge(); }
};

/**
 * @brief High-performance FastFlow mergesort with optimized memory management
 * and load balancing
 */
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data_ref,
                                     size_t num_threads) {
  if (data_ref.empty()) {
    return;
  }
  if (num_threads == 0) {
    num_threads = 1;
  }

  // Move data to temporary vector for processing
  std::vector<Record> data_to_sort = std::move(data_ref);
  data_ref.clear();

  // ====== Optimized Sorting Farm ======
  ff_farm sort_farm;

  // Create optimized workers
  std::vector<ff_node *> sort_workers;
  sort_workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    sort_workers.push_back(new HighPerformanceSortWorker());
  }
  sort_farm.add_workers(sort_workers);
  sort_farm.cleanup_workers(true);

  // Set up optimized emitter
  OptimizedSortEmitter *emitter =
      new OptimizedSortEmitter(data_to_sort, num_threads);
  sort_farm.add_emitter(emitter);
  sort_farm.cleanup_emitter(true);

  // Set up forwarding collector
  ForwardingCollector *collector = new ForwardingCollector();
  sort_farm.add_collector(collector);
  sort_farm.cleanup_collector(true);

  // Enable on-demand scheduling for better load balancing
  sort_farm.set_scheduling_ondemand();

  // ====== Optimized Final Merge Stage ======
  OptimizedFinalMerge final_merge(&data_ref);

  // ====== Pipeline Execution ======
  ff_pipeline pipeline;
  pipeline.add_stage(&sort_farm);
  pipeline.add_stage(&final_merge);

  if (pipeline.run_and_wait_end() < 0) {
    throw std::runtime_error("Pipeline execution failed");
  }
}

#ifdef TEST_MAIN
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "FastFlow Optimized Pipeline MergeSort\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Threads: " << config.num_threads << "\n\n";

  if (config.num_threads == 0) {
    std::cerr << "Warning: Number of threads is 0. Setting to 1.\n";
    config.num_threads = 1;
  }

  auto data_for_run = generate_data_vector(config.array_size,
                                           config.payload_size, config.pattern);
  size_t original_size = data_for_run.size();

  if (config.array_size <= 20) {
    std::vector<Record> data_copy = copy_records_vector(data_for_run);
    std::cout << "\nOriginal data (first few elements):\n";
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), data_copy.size());
         ++i) {
      std::cout << "Index " << i << ": key=" << data_copy[i].key << std::endl;
    }
  }

  Timer t("FF Optimized Pipeline MergeSort");

  ff_pipeline_two_farms_mergesort(data_for_run, config.num_threads);
  double ms = t.elapsed_ms();

  std::cout << "Time: " << ms << " ms\n";

  if (config.array_size <= 20) {
    std::cout << "\nData after sort (first few elements):\n";
    for (size_t i = 0;
         i < std::min(static_cast<size_t>(10), data_for_run.size()); ++i) {
      std::cout << "Index " << i << ": key=" << data_for_run[i].key
                << std::endl;
    }
  }

  std::cout << "Result vector size: " << data_for_run.size()
            << " (original: " << original_size << ")\n";

  if (config.validate) {
    if (!is_sorted_vector(data_for_run)) {
      std::cerr << "ERROR: Sort validation failed!\n";
      if (data_for_run.size() < 200 && !data_for_run.empty()) {
        std::cerr << "First few elements of failed sort: ";
        for (size_t i = 0;
             i < std::min(static_cast<size_t>(20), data_for_run.size()); ++i) {
          std::cerr << data_for_run[i].key << " ";
        }
        std::cerr << std::endl;
      }
      return 1;
    } else {
      std::cout << "Validation successful.\n";
    }
  }

  return 0;
}
#endif
