#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "ff_mergesort.hpp"
#include <algorithm>
#include <ff/farm.hpp>
#include <ff/ff.hpp>
#include <ff/pipeline.hpp>
#include <queue>

using namespace ff;

/**
 * @brief Splitter node - divides array into chunks
 */
class Splitter : public ff_node_t<SortTask> {
private:
  std::vector<std::unique_ptr<Record>> *data;
  size_t chunk_size;
  size_t num_chunks;

public:
  Splitter(std::vector<std::unique_ptr<Record>> *d, size_t nworkers) : data(d) {
    size_t n = data->size();
    num_chunks = std::min(nworkers * 4, std::max(size_t(1), n / 1000));
    chunk_size = (n + num_chunks - 1) / num_chunks;
  }

  SortTask *svc(SortTask *) {
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t start = i * chunk_size;
      size_t end = std::min(start + chunk_size, data->size());

      ff_send_out(new SortTask(data, start, end, i, num_chunks));
    }
    return EOS;
  }
};

/**
 * @brief Worker node - sorts individual chunks
 */
class SortWorker : public ff_node_t<SortTask, SortedChunk> {
public:
  SortedChunk *svc(SortTask *task) {
    // Sort chunk using std::sort for efficiency
    std::sort(task->data->begin() + task->start,
              task->data->begin() + task->end,
              [](const std::unique_ptr<Record> &a,
                 const std::unique_ptr<Record> &b) { return a->key < b->key; });

    auto result = new SortedChunk{task->start, task->end, task->chunk_id,
                                  task->total_chunks};
    delete task;
    return result;
  }
};

/**
 * @brief Merger node - performs k-way merge of sorted chunks
 */
class Merger : public ff_node_t<SortedChunk> {
private:
  std::vector<std::unique_ptr<Record>> *data;
  std::vector<SortedChunk *> chunks;
  size_t expected_chunks;

  struct ChunkIterator {
    size_t current;
    size_t end;
    size_t chunk_id;
  };

public:
  Merger(std::vector<std::unique_ptr<Record>> *d)
      : data(d), expected_chunks(0) {}

  SortedChunk *svc(SortedChunk *chunk) {
    if (expected_chunks == 0) {
      expected_chunks = chunk->total_chunks;
    }

    chunks.push_back(chunk);

    // Wait for all chunks
    if (chunks.size() < expected_chunks) {
      return GO_ON;
    }

    // Sort chunks by ID to ensure correct order
    std::sort(chunks.begin(), chunks.end(),
              [](const SortedChunk *a, const SortedChunk *b) {
                return a->chunk_id < b->chunk_id;
              });

    // Perform k-way merge
    k_way_merge();

    // Cleanup
    for (auto c : chunks)
      delete c;
    chunks.clear();

    return EOS;
  }

private:
  void k_way_merge() {
    std::vector<std::unique_ptr<Record>> temp;
    temp.reserve(data->size());

    // Min-heap for k-way merge
    auto cmp = [this](const ChunkIterator &a, const ChunkIterator &b) {
      return (*data)[a.current]->key > (*data)[b.current]->key;
    };
    std::priority_queue<ChunkIterator, std::vector<ChunkIterator>,
                        decltype(cmp)>
        heap(cmp);

    // Initialize heap with first element from each chunk
    for (const auto &chunk : chunks) {
      if (chunk->start < chunk->end) {
        heap.push({chunk->start, chunk->end, chunk->chunk_id});
      }
    }

    // Merge
    while (!heap.empty()) {
      ChunkIterator iter = heap.top();
      heap.pop();

      temp.push_back(std::move((*data)[iter.current]));

      iter.current++;
      if (iter.current < iter.end) {
        heap.push(iter);
      }
    }

    // Move back to original array
    *data = std::move(temp);
  }
};

/**
 * @brief FastFlow pipeline merge sort
 */
void ff_pipeline_mergesort(std::vector<std::unique_ptr<Record>> &data,
                           size_t nworkers) {
  Splitter splitter(&data, nworkers);
  Merger merger(&data);

  std::vector<std::unique_ptr<ff_node>> workers;
  for (size_t i = 0; i < nworkers; ++i) {
    workers.push_back(std::make_unique<SortWorker>());
  }

  ff_Farm<SortTask, SortedChunk> farm(std::move(workers));
  farm.add_emitter(splitter);
  farm.add_collector(merger);
  farm.set_scheduling_ondemand();

  if (farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Farm execution failed");
  }
}

// Main function for testing
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "FastFlow Pipeline MergeSort\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Threads: " << config.num_threads << "\n\n";

  auto data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  Timer t("FF Pipeline MergeSort");
  ff_pipeline_mergesort(data, config.num_threads);
  double ms = t.elapsed_ms();

  std::cout << "Time: " << ms << " ms\n";

  if (config.validate && !is_sorted(data)) {
    std::cerr << "ERROR: Sort validation failed!\n";
    return 1;
  }

  return 0;
}
