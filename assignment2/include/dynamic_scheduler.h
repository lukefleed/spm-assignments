#ifndef DYNAMIC_SCHEDULER_H
#define DYNAMIC_SCHEDULER_H

#include "common_types.h" // Includes Task, RangeResult, Config, ull
#include <atomic> // Potentially used by implementation or for global state
#include <condition_variable> // For TaskQueue synchronization
#include <deque> // Use std::deque for efficient push/pop at both ends (WorkStealingQueue)
#include <mutex> // For thread synchronization (std::mutex, std::lock_guard, std::unique_lock)
#include <optional> // For returning tasks that might not exist (std::optional<Task>)
#include <queue>  // For std::queue (used in the simple TaskQueue)
#include <vector> // For std::vector<RangeResult>, std::vector<WorkStealingQueue>

/**
 * @brief A simple thread-safe queue for distributing tasks among worker
 * threads. Implements a centralized queue using std::queue, std::mutex, and
 * std::condition_variable. Suitable for basic dynamic scheduling scenarios but
 * can become a bottleneck under high contention.
 */
class TaskQueue {
public:
  /**
   * @brief Pushes a task onto the queue. Thread-safe.
   * @param task The task to add.
   */
  void push(Task task);

  /**
   * @brief Pops a task from the queue. Blocks if the queue is empty until a
   * task is available or the queue is closed. Thread-safe.
   * @return An optional containing the task, or std::nullopt if the queue is
   * closed and empty.
   */
  std::optional<Task> pop();

  /**
   * @brief Closes the queue, preventing further pushes and notifying waiting
   * threads. Thread-safe.
   */
  void close();

private:
  std::queue<Task> queue_; /**< Underlying standard queue to hold tasks. */
  std::mutex
      mutex_; /**< Mutex to protect access to the queue and closed flag. */
  std::condition_variable cond_var_; /**< Condition variable for threads to wait
                                        for tasks or closure. */
  bool closed_ = false; /**< Flag indicating if the queue accepts new tasks. */
};

/**
 * @brief A thread-safe double-ended queue optimized for work-stealing
 * schedulers. Uses std::deque internally, allowing efficient push/pop from the
 * back (owner thread) and stealing from the front (thief threads). This
 * separation reduces contention compared to a single-ended queue under a
 * work-stealing pattern.
 */
class WorkStealingQueue {
public:
  WorkStealingQueue() = default;

  // Deleted copy/move constructors and assignment operators.
  // Queues contain mutexes which are non-copyable and non-movable.
  // Transferring ownership or copying a queue with threads potentially
  // operating on it is complex and generally undesirable. Each thread should
  // own its queue.
  WorkStealingQueue(const WorkStealingQueue &) = delete;
  WorkStealingQueue &operator=(const WorkStealingQueue &) = delete;
  WorkStealingQueue(WorkStealingQueue &&) = delete;
  WorkStealingQueue &operator=(WorkStealingQueue &&) = delete;

  /**
   * @brief Pushes a task onto the back (LIFO end) of the queue. Intended for
   * the owner thread. Thread-safe.
   * @param task The task to add.
   */
  void push(Task task);

  /**
   * @brief Pops a task from the back (LIFO end) of the queue. Intended for the
   * owner thread. Thread-safe.
   * @return An optional containing the task, or std::nullopt if the queue is
   * empty.
   */
  std::optional<Task> pop();

  /**
   * @brief Steals a task from the front (FIFO end) of the queue. Intended for
   * thief threads. Thread-safe.
   * @return An optional containing the stolen task, or std::nullopt if the
   * queue is empty.
   */
  std::optional<Task> steal();

  /**
   * @brief Checks if the queue is empty. Thread-safe.
   * @return True if the queue is empty, false otherwise.
   */
  bool empty() const;

  /**
   * @brief Returns the number of tasks currently in the queue. Thread-safe.
   * @return The current size of the queue.
   */
  size_t size() const;

private:
  std::deque<Task> queue_; /**< Underlying deque providing efficient front/back
                              operations. */
  /**
   * @brief Mutex protecting access to the deque. Marked 'mutable' so that
   *        const methods like `empty()` and `size()` can still acquire the lock
   *        to ensure thread-safe reading of the queue state without modifying
   *        the logical state of the WorkStealingQueue object itself.
   */
  mutable std::mutex mutex_;
};

/**
 * @brief Executes the Collatz computation using dynamic scheduling with a
 * single, centralized task queue.
 * @param config Program configuration (thread count, chunk size, ranges).
 * @param[out] results_out Vector where the results for each original range will
 * be stored.
 * @return True if execution completes successfully, false otherwise (e.g.,
 * invalid config).
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out);

/**
 * @brief Executes the Collatz computation using dynamic scheduling with
 * per-thread queues and work-stealing.
 * @param config Program configuration (thread count, chunk size, ranges).
 * @param[out] results_out Vector where the results for each original range will
 * be stored.
 * @return True if execution completes successfully, false otherwise (e.g.,
 * invalid config).
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out);

#endif // DYNAMIC_SCHEDULER_H
