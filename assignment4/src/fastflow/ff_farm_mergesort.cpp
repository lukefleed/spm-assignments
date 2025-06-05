#include "../common/record.hpp"
#include "../common/timer.hpp"
#include "../common/utils.hpp"
#include "ff_mergesort.hpp"
#include <algorithm>
#include <atomic>
#include <ff/farm.hpp>
#include <ff/ff.hpp>

using namespace ff;

/**
 * @brief Task for recursive merge sort using farm
 */
struct MergeTask {
  enum Type { SORT, MERGE };
  Type type;
  std::vector<std::unique_ptr<Record>> *data;
  size_t left;
  size_t right;
  size_t level;
  std::atomic<int> *pending_merges;

  MergeTask(Type t, std::vector<std::unique_ptr<Record>> *d, size_t l, size_t r,
            size_t lv, std::atomic<int> *pm)
      : type(t), data(d), left(l), right(r), level(lv), pending_merges(pm) {}
};

/**
 * @brief Emitter for recursive task generation
 */
class RecursiveEmitter : public ff_node_t<MergeTask> {
private:
  std::vector<std::unique_ptr<Record>> *data;
  size_t threshold;
  std::atomic<int> pending_merges{0};
  std::queue<MergeTask *> merge_queue;

public:
  RecursiveEmitter(std::vector<std::unique_ptr<Record>> *d, size_t nworkers)
      : data(d), threshold(std::max(size_t(1000), d->size() / (nworkers * 4))) {
  }

  MergeTask *svc(MergeTask *task) {
    if (task == nullptr) {
      // Initial task
      if (data->size() > threshold) {
        pending_merges.fetch_add(1);
        return new MergeTask(MergeTask::SORT, data, 0, data->size() - 1, 0,
                             &pending_merges);
      } else {
        // Small array, sort directly
        ff_send_out(new MergeTask(MergeTask::SORT, data, 0, data->size() - 1, 0,
                                  nullptr));
        return EOS;
      }
    }

    // Handle completed merge
    if (task->type == MergeTask::MERGE) {
      delete task;

      // Check if all merges completed
      if (pending_merges.fetch_sub(1) == 1) {
        return EOS;
      }
      return GO_ON;
    }

    delete task;
    return GO_ON;
  }
};

/**
 * @brief Worker for sorting and merging
 */
class SortMergeWorker : public ff_node_t<MergeTask> {
private:
  size_t threshold;

  void merge(std::vector<std::unique_ptr<Record>> &data, size_t left,
             size_t mid, size_t right) {
    std::vector<std::unique_ptr<Record>> temp;
    temp.reserve(right - left + 1);

    size_t i = left, j = mid + 1;

    // Merge
    while (i <= mid && j <= right) {
      if (data[i]->key <= data[j]->key) {
        temp.push_back(std::move(data[i++]));
      } else {
        temp.push_back(std::move(data[j++]));
      }
    }

    while (i <= mid)
      temp.push_back(std::move(data[i++]));
    while (j <= right)
      temp.push_back(std::move(data[j++]));

    // Move back
    for (size_t k = 0; k < temp.size(); ++k) {
      data[left + k] = std::move(temp[k]);
    }
  }

public:
  SortMergeWorker(size_t t) : threshold(t) {}

  MergeTask *svc(MergeTask *task) {
    if (task->type == MergeTask::SORT) {
      size_t size = task->right - task->left + 1;

      if (size <= threshold) {
        // Base case: sort directly
        std::sort(
            task->data->begin() + task->left,
            task->data->begin() + task->right + 1,
            [](const std::unique_ptr<Record> &a,
               const std::unique_ptr<Record> &b) { return a->key < b->key; });

        // Notify completion if needed
        if (task->pending_merges) {
          return new MergeTask(MergeTask::MERGE, nullptr, 0, 0, 0,
                               task->pending_merges);
        }
      } else {
        // Recursive case: split and send sub-tasks
        size_t mid = task->left + (task->right - task->left) / 2;

        if (task->pending_merges) {
          task->pending_merges->fetch_add(2);
        }

        ff_send_out(new MergeTask(MergeTask::SORT, task->data, task->left, mid,
                                  task->level + 1, task->pending_merges));
        ff_send_out(new MergeTask(MergeTask::SORT, task->data, mid + 1,
                                  task->right, task->level + 1,
                                  task->pending_merges));

        // Schedule merge after subtasks complete
        ff_send_out(new MergeTask(MergeTask::MERGE, task->data, task->left,
                                  task->right, task->level,
                                  task->pending_merges));
      }
    } else {
      // Merge task
      size_t mid = task->left + (task->right - task->left) / 2;
      merge(*task->data, task->left, mid, task->right);

      if (task->pending_merges) {
        return new MergeTask(MergeTask::MERGE, nullptr, 0, 0, 0,
                             task->pending_merges);
      }
    }

    delete task;
    return GO_ON;
  }
};

/**
 * @brief Farm-based merge sort with feedback
 */
void ff_farm_mergesort(std::vector<std::unique_ptr<Record>> &data,
                       size_t nworkers) {
  size_t threshold = std::max(size_t(1000), data.size() / (nworkers * 4));

  RecursiveEmitter emitter(&data, nworkers);

  std::vector<std::unique_ptr<ff_node>> workers;
  for (size_t i = 0; i < nworkers; ++i) {
    workers.push_back(std::make_unique<SortMergeWorker>(threshold));
  }

  ff_Farm<MergeTask> farm(std::move(workers));
  farm.add_emitter(emitter);
  farm.wrap_around(); // Enable feedback for recursive tasks

  if (farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Farm execution failed");
  }
}
