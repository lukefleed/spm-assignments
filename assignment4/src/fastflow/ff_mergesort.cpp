#include "../common/record.hpp"
#include <algorithm>
#include <deque>
#include <ff/ff.hpp>
#include <functional>
#include <memory>
#include <vector>

using namespace ff;

/**
 * @struct SortChunk
 * @brief Represents a chunk of data.
 */
struct SortChunk {
  std::vector<Record> data;
};

/**
 * @class SortEmitter
 * @brief Emitter for the sorting farm. Chunks the initial dataset.
 */
class SortEmitter : public ff_node {
private:
  std::vector<Record> &original_data;
  size_t chunk_size;
  size_t num_chunks;
  size_t chunks_sent = 0;

public:
  SortEmitter(std::vector<Record> &data, size_t workers)
      : original_data(data), num_chunks(0), chunk_size(0) {
    if (!original_data.empty() && workers > 0) {
      // Over-decompose for better load balancing.
      num_chunks = workers * 4;
      chunk_size = (original_data.size() + num_chunks - 1) / num_chunks;
    }
  }

  void *svc(void * /*task*/) override {
    if (chunks_sent >= num_chunks || chunk_size == 0) {
      return EOS;
    }

    size_t start = chunks_sent * chunk_size;
    if (start >= original_data.size()) {
      return EOS;
    }
    size_t end = std::min(start + chunk_size, original_data.size());

    auto *chunk = new SortChunk();
    chunk->data.assign(std::make_move_iterator(original_data.begin() + start),
                       std::make_move_iterator(original_data.begin() + end));
    chunks_sent++;
    return chunk;
  }
};

/**
 * @class SortWorker
 * @brief Worker node for the sorting farm. Sorts individual chunks.
 */
class SortWorker : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *chunk) override {
    std::sort(chunk->data.begin(), chunk->data.end());
    return chunk;
  }
};

/**
 * @class ForwardingCollector
 * @brief A minimal collector that provides an output channel for the farm.
 */
class ForwardingCollector : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *task) override { return task; }
};

/**
 * @class FinalMergeNode
 * @brief A stateful node that collects all chunks and merges them in svc_end().
 *
 * This node's svc() method only buffers incoming chunks. The final, guaranteed-
 * to-be-correct iterative merge is executed in svc_end(), which FastFlow calls
 * only after all upstream tasks and EOS signals have been processed.
 */
class FinalMergeNode : public ff_node_t<SortChunk, void> {
private:
  std::vector<Record> *final_result_ptr;
  std::deque<SortChunk *> chunks_buffer;

  void merge_two_vectors(std::vector<Record> &left, std::vector<Record> &right,
                         std::vector<Record> &result) {
    result.reserve(left.size() + right.size());
    auto l_it = left.begin(), r_it = right.begin();
    while (l_it != left.end() && r_it != right.end()) {
      if (*l_it <= *r_it) {
        result.push_back(std::move(*l_it++));
      } else {
        result.push_back(std::move(*r_it++));
      }
    }
    result.insert(result.end(), std::make_move_iterator(l_it),
                  std::make_move_iterator(left.end()));
    result.insert(result.end(), std::make_move_iterator(r_it),
                  std::make_move_iterator(right.end()));
  }

public:
  explicit FinalMergeNode(std::vector<Record> *result_vec)
      : final_result_ptr(result_vec) {}

  ~FinalMergeNode() {
    for (auto *chunk : chunks_buffer) {
      delete chunk;
    }
  }

  void *svc(SortChunk *chunk) override {
    if (chunk) {
      chunks_buffer.push_back(chunk);
    }
    // No termination logic here. Just buffer chunks.
    return GO_ON;
  }

  void svc_end() override {
    // This is the safe place for the final merge.
    while (chunks_buffer.size() > 1) {
      SortChunk *left = chunks_buffer.front();
      chunks_buffer.pop_front();
      SortChunk *right = chunks_buffer.front();
      chunks_buffer.pop_front();

      auto *merged_chunk = new SortChunk();
      merge_two_vectors(left->data, right->data, merged_chunk->data);
      chunks_buffer.push_back(merged_chunk);

      delete left;
      delete right;
    }

    if (chunks_buffer.size() == 1) {
      if (final_result_ptr) {
        *final_result_ptr = std::move(chunks_buffer.front()->data);
      }
      delete chunks_buffer.front();
      chunks_buffer.pop_front();
    } else if (final_result_ptr) {
      final_result_ptr->clear();
    }
  }
};

/**
 * @brief Performs a parallel merge sort using a single farm and a final merge
 * node.
 * @param data_ref A reference to the vector of Records to be sorted.
 * @param num_threads The total number of threads to be used by FastFlow.
 */
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data_ref,
                                     size_t num_threads) {
  if (data_ref.size() <= 1) {
    return;
  }
  if (num_threads == 0) {
    num_threads = 1;
  }

  // The SortEmitter needs a reference to the data, so we can't move it yet.
  // The original data_ref will be cleared and used for the final result.
  std::vector<Record> data_to_sort = std::move(data_ref);
  data_ref.clear();

  // ====== 1. Sorting Farm ======
  ff_farm sort_farm;
  std::vector<ff_node *> sort_workers_raw;
  sort_workers_raw.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    sort_workers_raw.push_back(new SortWorker());
  }
  sort_farm.add_workers(sort_workers_raw);
  sort_farm.cleanup_workers(true);

  SortEmitter *se = new SortEmitter(data_to_sort, num_threads);
  sort_farm.add_emitter(se);
  sort_farm.cleanup_emitter(true);

  ForwardingCollector *fc = new ForwardingCollector();
  sort_farm.add_collector(fc);
  sort_farm.cleanup_collector(true);

  // Using on-demand scheduling is generally better for load balancing
  sort_farm.set_scheduling_ondemand();

  // ====== 2. Final Merge Node ======
  FinalMergeNode merge_node(&data_ref);

  // ====== Pipeline Execution ======
  ff_pipeline pipeline;
  pipeline.add_stage(&sort_farm);
  pipeline.add_stage(&merge_node);

  if (pipeline.run_and_wait_end() < 0) {
    throw std::runtime_error("Pipeline execution failed");
  }
}
