#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "ff_mergesort.hpp"
#include <algorithm>
#include <ff/all2all.hpp>
#include <ff/ff.hpp>
#include <numeric>

using namespace ff;

/**
 * @brief Task for All-to-All pattern
 */
struct A2ATask {
  enum Phase { SAMPLE, PARTITION, SORT, DONE };
  Phase phase;
  std::vector<std::unique_ptr<Record>> *data;
  size_t start;
  size_t end;
  std::vector<unsigned long> samples;
  std::vector<unsigned long> pivots;
  size_t worker_id;

  A2ATask(Phase p = SAMPLE)
      : phase(p), data(nullptr), start(0), end(0), worker_id(0) {}
};

/**
 * @brief First set of workers - sampling and partitioning
 */
class PartitionWorker : public ff_node_t<A2ATask> {
private:
  size_t worker_id;
  size_t total_workers;
  std::vector<std::unique_ptr<Record>> *global_data;
  size_t chunk_start, chunk_end;
  std::vector<unsigned long> all_samples;
  std::vector<unsigned long> pivots;

public:
  PartitionWorker(size_t id, size_t total,
                  std::vector<std::unique_ptr<Record>> *data)
      : worker_id(id), total_workers(total), global_data(data) {
    size_t chunk_size = data->size() / total_workers;
    chunk_start = id * chunk_size;
    chunk_end =
        (id == total_workers - 1) ? data->size() : (id + 1) * chunk_size;
  }

  A2ATask *svc(A2ATask *task) {
    if (task == nullptr || task->phase == A2ATask::SAMPLE) {
      // Phase 1: Sample elements for pivot selection
      size_t sample_size =
          std::min(size_t(100), (chunk_end - chunk_start) / 10);
      std::vector<unsigned long> samples;

      for (size_t i = 0; i < sample_size; ++i) {
        size_t idx =
            chunk_start + (i * (chunk_end - chunk_start)) / sample_size;
        samples.push_back((*global_data)[idx]->key);
      }

      auto result = new A2ATask(A2ATask::PARTITION);
      result->samples = std::move(samples);
      result->worker_id = worker_id;

      delete task;
      return result;
    }

    if (task->phase == A2ATask::PARTITION) {
      // Collect samples from all workers
      all_samples.insert(all_samples.end(), task->samples.begin(),
                         task->samples.end());

      if (all_samples.size() >= total_workers * 10) {
        // Select pivots
        std::sort(all_samples.begin(), all_samples.end());
        pivots.clear();

        for (size_t i = 1; i < total_workers; ++i) {
          pivots.push_back(all_samples[i * all_samples.size() / total_workers]);
        }

        // Partition local data and send to appropriate workers
        std::vector<std::vector<size_t>> partitions(total_workers);

        for (size_t i = chunk_start; i < chunk_end; ++i) {
          unsigned long key = (*global_data)[i]->key;
          size_t dest = std::lower_bound(pivots.begin(), pivots.end(), key) -
                        pivots.begin();
          partitions[dest].push_back(i);
        }

        // Send partitions to second set workers
        for (size_t dest = 0; dest < total_workers; ++dest) {
          auto out_task = new A2ATask(A2ATask::SORT);
          out_task->data = global_data;
          out_task->worker_id = dest;
          // In real implementation, would send indices
          ff_send_out_to(out_task, dest);
        }
      }

      delete task;
      return GO_ON;
    }

    return EOS;
  }
};

/**
 * @brief Second set of workers - sorting partitions
 */
class SortWorker : public ff_node_t<A2ATask> {
private:
  size_t worker_id;
  std::vector<size_t> my_indices;
  std::vector<std::unique_ptr<Record>> *global_data;

public:
  SortWorker(size_t id) : worker_id(id), global_data(nullptr) {}

  A2ATask *svc(A2ATask *task) {
    if (task->phase == A2ATask::SORT) {
      if (global_data == nullptr) {
        global_data = task->data;
      }

      // Collect indices from all partition workers
      // In real implementation, would receive actual indices

      // For now, simulate by sorting a portion
      size_t total_workers = ff_node::get_num_inchannels();
      size_t chunk_size = global_data->size() / total_workers;
      size_t start = worker_id * chunk_size;
      size_t end = (worker_id == total_workers - 1)
                       ? global_data->size()
                       : (worker_id + 1) * chunk_size;

      std::sort(
          global_data->begin() + start, global_data->begin() + end,
          [](const std::unique_ptr<Record> &a,
             const std::unique_ptr<Record> &b) { return a->key < b->key; });

      delete task;
      return new A2ATask(A2ATask::DONE);
    }

    delete task;
    return EOS;
  }
};

/**
 * @brief All-to-All based merge sort (simplified version)
 */
void ff_all2all_mergesort(std::vector<std::unique_ptr<Record>> &data,
                          size_t nworkers) {
  std::vector<ff_node *> left_workers;
  std::vector<ff_node *> right_workers;

  // Create partition workers
  for (size_t i = 0; i < nworkers; ++i) {
    left_workers.push_back(new PartitionWorker(i, nworkers, &data));
  }

  // Create sort workers
  for (size_t i = 0; i < nworkers; ++i) {
    right_workers.push_back(new SortWorker(i));
  }

  ff_a2a a2a;
  a2a.add_firstset(left_workers, 0, true);
  a2a.add_secondset(right_workers, true);

  if (a2a.run_and_wait_end() < 0) {
    throw std::runtime_error("All-to-All execution failed");
  }

  // Note: This is a simplified version. A full implementation would need
  // proper data redistribution and final merging phase
}

// Main function
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "FastFlow All-to-All MergeSort (Simplified)\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Threads: " << config.num_threads << "\n\n";

  auto data =
      generate_data(config.array_size, config.payload_size, config.pattern);

  Timer t("FF All2All MergeSort");
  ff_all2all_mergesort(data, config.num_threads);
  double ms = t.elapsed_ms();

  std::cout << "Time: " << ms << " ms\n";
  std::cout << "Note: This is a simplified demonstration\n";

  return 0;
}
