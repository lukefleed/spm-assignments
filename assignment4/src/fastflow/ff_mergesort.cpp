#include "../common/utils.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
#include <queue>

using namespace ff;

// ====== STRUCTURES FOR PIPELINE ======

struct SortChunk {
  std::vector<Record> data;
  size_t chunk_id;
  size_t level; // For merge levels

  SortChunk(std::vector<Record> d, size_t id, size_t lvl = 0)
      : data(std::move(d)), chunk_id(id), level(lvl) {}
};

struct MergePair {
  std::vector<Record> left;
  std::vector<Record> right;
  size_t result_id;
  size_t level;

  MergePair(std::vector<Record> l, std::vector<Record> r, size_t id, size_t lvl)
      : left(std::move(l)), right(std::move(r)), result_id(id), level(lvl) {}
};

// ====== FIRST FARM: SORTING ======

class SortEmitter : public ff_node {
private:
  std::vector<Record> &original_data;
  size_t num_workers;
  size_t chunk_size;
  size_t current_chunk;
  size_t total_chunks;

public:
  SortEmitter(std::vector<Record> &data, size_t workers)
      : original_data(data), num_workers(workers), current_chunk(0) {
    chunk_size = (data.size() + workers - 1) / workers;
    total_chunks = (data.size() + chunk_size - 1) / chunk_size;
  }

  void *svc(void *) override {
    if (current_chunk >= total_chunks) {
      return EOS;
    }

    size_t start = current_chunk * chunk_size;
    size_t end = std::min(start + chunk_size, original_data.size());

    // Create chunk data by moving elements
    std::vector<Record> chunk_data;
    chunk_data.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      chunk_data.push_back(std::move(original_data[i]));
    }

    auto chunk = new SortChunk(std::move(chunk_data), current_chunk);
    current_chunk++;
    return chunk;
  }
};

class SortWorker : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *chunk) override {
    if (chunk == nullptr)
      return EOS;

    // Sort the chunk
    std::sort(chunk->data.begin(), chunk->data.end());

    return chunk;
  }
};

class SortCollector : public ff_node_t<SortChunk, SortChunk> {
private:
  std::vector<SortChunk *> collected_chunks;
  size_t expected_chunks;
  size_t sent_chunks;
  bool all_collected;

public:
  SortCollector(size_t expected)
      : expected_chunks(expected), sent_chunks(0), all_collected(false) {}

  SortChunk *svc(SortChunk *chunk) override {
    if (chunk == nullptr) {
      // End of input - check if we have more chunks to send
      if (sent_chunks < collected_chunks.size()) {
        return collected_chunks[sent_chunks++];
      }
      return EOS;
    }

    // Collect incoming chunk
    collected_chunks.push_back(chunk);

    // If this is the last chunk we were waiting for
    if (collected_chunks.size() == expected_chunks) {
      all_collected = true;
      // Start sending chunks immediately
      if (sent_chunks < collected_chunks.size()) {
        return collected_chunks[sent_chunks++];
      }
    }

    // If we have already collected all chunks, continue sending
    if (all_collected && sent_chunks < collected_chunks.size()) {
      return collected_chunks[sent_chunks++];
    }

    return GO_ON;
  }
};

// ====== SECOND FARM: MERGING ======

class MergeEmitter : public ff_node_t<SortChunk, MergePair> {
private:
  std::vector<SortChunk *> pending_chunks;
  size_t pair_id;

public:
  MergeEmitter() : pair_id(0) {}

  MergePair *svc(SortChunk *chunk) override {
    if (chunk == nullptr) {
      // Process remaining chunks
      if (pending_chunks.size() == 1) {
        // Single chunk left, send as final result
        auto *final_chunk = pending_chunks[0];
        pending_chunks.clear();

        // Convert to MergePair with empty right side
        auto pair =
            new MergePair(std::move(final_chunk->data), std::vector<Record>(),
                          pair_id++, final_chunk->level);
        delete final_chunk;
        return pair;
      }
      return EOS;
    }

    pending_chunks.push_back(chunk);

    // When we have a pair, send for merging
    if (pending_chunks.size() >= 2) {
      auto *left = pending_chunks[0];
      auto *right = pending_chunks[1];
      pending_chunks.erase(pending_chunks.begin(), pending_chunks.begin() + 2);

      auto pair =
          new MergePair(std::move(left->data), std::move(right->data),
                        pair_id++, std::max(left->level, right->level) + 1);

      delete left;
      delete right;
      return pair;
    }

    return GO_ON;
  }
};

class MergeWorker : public ff_node_t<MergePair, SortChunk> {
public:
  SortChunk *svc(MergePair *pair) override {
    if (pair == nullptr)
      return EOS;

    std::vector<Record> merged;

    if (pair->right.empty()) {
      // Single chunk case
      merged = std::move(pair->left);
    } else {
      // Merge two chunks
      merge_two_vectors(pair->left, pair->right, merged);
    }

    auto result =
        new SortChunk(std::move(merged), pair->result_id, pair->level);
    delete pair;
    return result;
  }

private:
  void merge_two_vectors(std::vector<Record> &left, std::vector<Record> &right,
                         std::vector<Record> &result) {
    result.reserve(left.size() + right.size());

    size_t i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
      if (left[i] <= right[j]) {
        result.push_back(std::move(left[i++]));
      } else {
        result.push_back(std::move(right[j++]));
      }
    }

    while (i < left.size()) {
      result.push_back(std::move(left[i++]));
    }

    while (j < right.size()) {
      result.push_back(std::move(right[j++]));
    }
  }
};

class MergeCollector : public ff_node_t<SortChunk, void> {
private:
  std::vector<Record> *final_result;
  std::vector<SortChunk *> collected_chunks;
  size_t expected_chunks;

public:
  MergeCollector(std::vector<Record> *result, size_t expected)
      : final_result(result), expected_chunks(expected) {}

  void *svc(SortChunk *chunk) override {
    if (chunk == nullptr)
      return EOS;

    collected_chunks.push_back(chunk);

    // When all chunks are collected, do k-way merge
    if (collected_chunks.size() == expected_chunks) {
      k_way_merge();

      // Cleanup chunks
      for (auto *c : collected_chunks) {
        delete c;
      }
    }

    return GO_ON;
  }

private:
  void k_way_merge() {
    using Iterator = std::vector<Record>::iterator;

    // Priority queue for k-way merge
    auto compare = [](const std::pair<Iterator, Iterator> &a,
                      const std::pair<Iterator, Iterator> &b) {
      return *a.first > *b.first; // Min heap
    };

    std::priority_queue<std::pair<Iterator, Iterator>,
                        std::vector<std::pair<Iterator, Iterator>>,
                        decltype(compare)>
        pq(compare);

    // Calculate total size
    size_t total_size = 0;
    for (const auto &chunk : collected_chunks) {
      total_size += chunk->data.size();
    }

    final_result->clear();
    final_result->reserve(total_size);

    // Initialize with first element from each chunk
    for (auto &chunk : collected_chunks) {
      if (!chunk->data.empty()) {
        pq.push({chunk->data.begin(), chunk->data.end()});
      }
    }

    // Merge all chunks
    while (!pq.empty()) {
      auto [current, end] = pq.top();
      pq.pop();

      final_result->push_back(std::move(*current));

      ++current;
      if (current != end) {
        pq.push({current, end});
      }
    }
  }
};

// ====== MAIN PIPELINE FUNCTION ======

void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     size_t num_threads) {
  if (data.empty())
    return;

  size_t sort_workers = num_threads / 2;
  size_t merge_workers = num_threads - sort_workers;
  if (sort_workers == 0)
    sort_workers = 1;
  if (merge_workers == 0)
    merge_workers = 1;

  // Create first farm for sorting (NO COLLECTOR!)
  std::vector<ff_node *> sort_workers_vec;
  for (size_t i = 0; i < sort_workers; i++) {
    sort_workers_vec.push_back(new SortWorker());
  }

  ff_farm sort_farm;
  auto sort_emitter = new SortEmitter(data, sort_workers);

  sort_farm.add_emitter(sort_emitter);
  sort_farm.add_workers(sort_workers_vec);
  // NO COLLECTOR - let chunks flow directly to next stage

  // Create second farm for merging
  std::vector<ff_node *> merge_workers_vec;
  for (size_t i = 0; i < merge_workers; i++) {
    merge_workers_vec.push_back(new MergeWorker());
  }

  ff_farm merge_farm;
  auto merge_emitter = new MergeEmitter();
  auto merge_collector =
      new MergeCollector(&data, sort_workers); // Pass expected chunks

  merge_farm.add_emitter(merge_emitter);
  merge_farm.add_workers(merge_workers_vec);
  merge_farm.add_collector(merge_collector);

  // Create pipeline
  ff_pipeline pipeline;
  pipeline.add_stage(&sort_farm);
  pipeline.add_stage(&merge_farm);

  // Execute
  if (pipeline.run_and_wait_end() < 0) {
    throw std::runtime_error("Pipeline execution failed");
  }

  // Cleanup
  delete sort_emitter;
  for (auto *worker : sort_workers_vec) {
    delete worker;
  }

  delete merge_emitter;
  delete merge_collector;
  for (auto *worker : merge_workers_vec) {
    delete worker;
  }
}

#ifdef TEST_MAIN
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "FastFlow Pipeline Two Farms MergeSort\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Threads: " << config.num_threads << "\n\n";

  auto data = generate_data_vector(config.array_size, config.payload_size,
                                   config.pattern);

  Timer t("FF Pipeline Two Farms MergeSort");
  ff_pipeline_two_farms_mergesort(data, config.num_threads);
  double ms = t.elapsed_ms();

  std::cout << "Time: " << ms << " ms\n";

  if (config.validate && !is_sorted_vector(data)) {
    std::cerr << "ERROR: Sort validation failed!\n";
    return 1;
  }

  return 0;
}
#endif
