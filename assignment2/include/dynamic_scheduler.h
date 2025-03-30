#ifndef DYNAMIC_SCHEDULER_H
#define DYNAMIC_SCHEDULER_H

#include "common_types.h"
#include <atomic>
#include <condition_variable>
#include <deque> // Use deque for work-stealing queue
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

// --- TaskQueue ---
class TaskQueue {
public:
  void push(Task task);
  std::optional<Task> pop();
  void close();

private:
  std::queue<Task> queue_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  bool closed_ = false;
};

// --- Work-Stealing Queue ---
class WorkStealingQueue {
public:
  WorkStealingQueue() = default;
  // Non-copyable and non-movable to avoid complexities with mutexes.
  WorkStealingQueue(const WorkStealingQueue &) = delete;
  WorkStealingQueue &operator=(const WorkStealingQueue &) = delete;
  WorkStealingQueue(WorkStealingQueue &&) = delete;
  WorkStealingQueue &operator=(WorkStealingQueue &&) = delete;

  // Add a task to the local queue.
  void push(Task task);

  // Attempt to pop a task from the local queue.
  std::optional<Task> pop();

  // Attempt to steal a task from another queue.
  std::optional<Task> steal();

  // Check if the queue is empty.
  bool empty() const;

  // Get the size of the queue.
  size_t size() const;

private:
  std::deque<Task> queue_;
  mutable std::mutex
      mutex_; // Mutex needs to be mutable for const methods like empty()
};

/**
 * @brief Executes the computation using dynamic scheduling with a single task
 * queue.
 * @param config Program configuration.
 * @param results_out Vector to store the results.
 * @return True on success, false otherwise.
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out);

/**
 * @brief Executes the computation using dynamic scheduling with work-stealing.
 * @param config Program configuration.
 * @param results_out Vector to store the results.
 * @return True on success, false otherwise.
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out);

#endif // DYNAMIC_SCHEDULER_H
