#include "dynamic_scheduler.h"
#include "collatz.h"
#include <atomic>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <thread>
#include <vector>

/**
 * @brief Pushes a task onto the queue.
 *
 * @param task The task to be added to the queue.
 */
void TaskQueue::push(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return; // Do not add tasks to a closed queue.
    }
    queue_.push(std::move(task));
  }
  cond_var_.notify_one(); // Notify one waiting thread that a task is available.
}

/**
 * @brief Pops a task from the queue.
 *
 * @return An optional containing the task if the queue is not empty,
 *         std::nullopt otherwise.
 */
std::optional<Task> TaskQueue::pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.wait(lock, [this] {
    return !queue_.empty() ||
           closed_; // Wait until queue is not empty or is closed.
  });
  if (queue_.empty()) {
    return std::nullopt; // Return nullopt if queue is empty.
  }
  Task task = std::move(queue_.front());
  queue_.pop();
  return task;
}

/**
 * @brief Closes the queue to prevent further tasks from being added.
 */
void TaskQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true; // Set the closed flag to prevent new tasks.
  }
  cond_var_.notify_all(); // Notify all waiting threads to exit.
}

/**
 * @brief Pushes a task onto the back of the queue.
 *
 * @param task The task to be added to the queue.
 */
void WorkStealingQueue::push(Task task) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(std::move(task));
}

/**
 * @brief Pops a task from the back of the queue (LIFO).
 *
 * @return An optional containing the task if the queue is not empty,
 *         std::nullopt otherwise.
 */
std::optional<Task> WorkStealingQueue::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  Task task = std::move(queue_.back());
  queue_.pop_back();
  return task;
}

/**
 * @brief Steals a task from the front of the queue (FIFO).
 *
 * @return An optional containing the stolen task if the queue is not empty,
 *         std::nullopt otherwise.
 */
std::optional<Task> WorkStealingQueue::steal() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  Task task = std::move(queue_.front());
  queue_.pop_front();
  return task;
}

/**
 * @brief Checks if the queue is empty.
 *
 * @return True if the queue is empty, false otherwise.
 */
bool WorkStealingQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

/**
 * @brief Returns the number of tasks in the queue.
 *
 * @return The number of tasks in the queue.
 */
size_t WorkStealingQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

/**
 * @brief Global counter for pending tasks in the work-stealing scheduler.
 */
std::atomic<size_t> g_pending_tasks_ws{0};

/**
 * @brief Flag indicating whether all initial tasks have been submitted to the
 * work-stealing scheduler.
 */
std::atomic<bool> g_all_tasks_submitted_ws{false};

/**
 * @brief Worker thread function for the dynamic scheduler.
 *
 * This function continuously retrieves tasks from the queue and processes them
 * until the queue is closed.
 *
 * @param thread_id The ID of the thread.
 * @param queue The task queue to retrieve tasks from.
 * @param results_out A vector to store the results of the computations.
 */
void dynamic_worker(int thread_id [[maybe_unused]], TaskQueue &queue,
                    std::vector<RangeResult> &results_out) {
  while (true) {
    std::optional<Task> task_opt = queue.pop();
    if (!task_opt) {
      break; // Exit loop if no task is available.
    }
    Task task = *task_opt;
    ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);
    ull current_max = results_out[task.original_range_index].max_steps.load(
        std::memory_order_relaxed);
    while (local_max_steps > current_max) {
      // Atomically update the maximum steps if the local value is greater.
      if (results_out[task.original_range_index]
              .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
        break; // Exit loop if update was successful.
      }
      std::this_thread::yield(); // Yield the thread to avoid busy-waiting.
    }
  }
}

/**
 * @brief Executes the Collatz computation using a dynamic task queue.
 *
 * This function creates a task queue, spawns worker threads, and distributes
 * the computation of Collatz sequences across the specified ranges.
 *
 * @param config The configuration parameters for the computation.
 * @param results_out A vector to store the results of the computations.
 * @return True if the computation was successful, false otherwise.
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out) {
  if (config.num_threads <= 0 || config.chunk_size == 0)
    return false; // Ensure valid configuration.

  TaskQueue task_queue;
  std::vector<std::thread> threads;

  results_out.clear();
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r); // Initialize result storage for each range.
  }

  threads.reserve(config.num_threads);
  for (int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(dynamic_worker, i, std::ref(task_queue),
                         std::ref(results_out)); // Create worker threads.
  }

  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue; // Skip invalid ranges.

    ull current_start = range.start;
    while (current_start <= range.end) {
      ull current_chunk_end;
      // Avoid overflow when calculating chunk end.
      if (current_start >
          std::numeric_limits<ull>::max() - (config.chunk_size - 1)) {
        current_chunk_end = range.end;
      } else {
        current_chunk_end = current_start + config.chunk_size - 1;
      }
      ull current_end = std::min(range.end, current_chunk_end);
      task_queue.push(
          {current_start, current_end, i}); // Push task to the queue.
      if (current_end == range.end)
        break;
      current_start = current_end + 1;
    }
  }

  task_queue.close(); // Signal no more tasks will be added.

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join(); // Wait for all threads to finish.
    }
  }

  return true;
}

/**
 * @brief Worker thread function for the work-stealing dynamic scheduler.
 *
 * Each worker has a preferred queue. If empty, it attempts to steal from other
 * queues. To avoid unproductive stealing attempts, a backoff mechanism is used.
 *
 * @param thread_id The ID of the thread.
 * @param num_threads The total number of threads.
 * @param queues A vector of all work-stealing queues.
 * @param results_out A vector to store the results of the computations.
 */
void dynamic_work_stealing_worker(
    int thread_id, int num_threads,
    std::vector<WorkStealingQueue> &queues, // Vector of all queues
    std::vector<RangeResult> &results_out) {
  // Thread-local vectors to track failed stealing attempts and backoff times
  thread_local std::vector<int> failed_attempts(num_threads, 0);
  thread_local std::vector<int> backoff_times(num_threads, 0);

  while (true) {
    std::optional<Task> task_opt;

    // 1. Try to pop from own queue
    task_opt = queues[thread_id].pop();

    // 2. If own queue is empty, try to steal
    if (!task_opt) {
      // Iterate through other queues to attempt stealing, skipping those in
      // backoff
      int current_victim = (thread_id + 1) % num_threads;
      for (int i = 0; i < num_threads - 1; i++) {
        if (current_victim != thread_id && backoff_times[current_victim] == 0) {
          task_opt = queues[current_victim].steal();
          if (task_opt) {
            failed_attempts[current_victim] =
                0; // Reset failed attempts on success
            break;
          } else {
            // Increase backoff on failure
            failed_attempts[current_victim]++;
            if (failed_attempts[current_victim] > 3) {
              // Implement backoff: if stealing fails multiple times, increase
              // backoff time
              backoff_times[current_victim] =
                  failed_attempts[current_victim] * 5;
            }
          }
        }

        // Decrease backoff counters
        for (int j = 0; j < num_threads; j++) {
          if (backoff_times[j] > 0)
            backoff_times[j]--;
        }

        current_victim = (current_victim + 1) % num_threads;
      }
    }

    // 3. Process task if found
    if (task_opt) {
      Task task = *task_opt;
      ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

      // Atomically update the maximum steps for this range
      ull current_max = results_out[task.original_range_index].max_steps.load(
          std::memory_order_relaxed);
      while (local_max_steps > current_max) {
        if (results_out[task.original_range_index]
                .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                                 std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
          break; // Update successful
        }
        // No yield here, CAS loop handles contention implicitly
      }
      // Signal task completion *after* processing
      g_pending_tasks_ws.fetch_sub(1, std::memory_order_release);
    } else {
      // 4. No work found (local or stolen from one victim) -> Check termination
      //    Load with acquire semantics to synchronize with release operations
      //    (fetch_sub and store on g_all_tasks_submitted_ws)
      if (g_all_tasks_submitted_ws.load(std::memory_order_acquire) &&
          g_pending_tasks_ws.load(std::memory_order_acquire) == 0) {
        // Potential termination condition met.
        // To be safer, we could double-check all queues are empty,
        // but relying on the counter is sufficient if task
        // generation dynamics are simple (like here).

        // bool potentially_done = true;
        // Optional: Add a check across *all* queues for emptiness for
        // robustness if (!potentially_done) continue; // Skip break if check
        // fails

        // Let's try a stronger check: re-check pending tasks after ensuring
        // visibility
        std::atomic_thread_fence(
            std::memory_order_seq_cst); // Ensure prior loads/stores are visible
        if (g_pending_tasks_ws.load(std::memory_order_relaxed) == 0) {
          break; // Exit worker loop
        }
      }

      // If not terminating, yield to prevent busy-spinning when idle
      std::this_thread::yield();
    }
  } // end while(true)
}

/**
 * @brief Executes the Collatz computation using dynamic scheduling with work
 * stealing.
 *
 * This function creates a set of work-stealing queues, spawns worker threads,
 * and distributes the computation of Collatz sequences across the specified
 * ranges.
 *
 * @param config The configuration parameters for the computation.
 * @param results_out A vector to store the results of the computations.
 * @return True if the computation was successful, false otherwise.
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out) {
  if (config.num_threads <= 0 || config.chunk_size == 0)
    return false; // Validate configuration parameters.

  // Initialize shared state
  g_pending_tasks_ws.store(0, std::memory_order_relaxed);
  g_all_tasks_submitted_ws.store(false, std::memory_order_relaxed);

  // Create per-thread queues
  std::vector<WorkStealingQueue> queues(config.num_threads);
  std::vector<std::thread> threads;

  // Initialize results vector
  results_out.clear();
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r); // Initialize result storage for each range.
  }

  // Create and start worker threads
  threads.reserve(config.num_threads);
  for (int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(dynamic_work_stealing_worker, i, config.num_threads,
                         std::ref(queues),
                         std::ref(results_out)); // Create worker threads.
  }

  // Main thread divides work into tasks and distributes them (round-robin)
  size_t tasks_created = 0;
  size_t current_queue_idx = 0;
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue; // Skip invalid ranges.

    ull current_start = range.start;
    while (current_start <= range.end) {
      ull current_chunk_end;
      // Avoid overflow when calculating chunk end.
      if (current_start >
          std::numeric_limits<ull>::max() - (config.chunk_size - 1)) {
        current_chunk_end = range.end; // Avoid overflow
      } else {
        current_chunk_end = current_start + config.chunk_size - 1;
      }
      ull current_end = std::min(range.end, current_chunk_end);

      // Increment pending task count *before* pushing
      // Relaxed is fine, synchronization happens via queue/worker later
      g_pending_tasks_ws.fetch_add(1, std::memory_order_relaxed);
      tasks_created++;

      // Push task to one of the queues (round-robin)
      queues[current_queue_idx].push({current_start, current_end, i});
      current_queue_idx = (current_queue_idx + 1) % config.num_threads;

      if (current_end == range.end)
        break;
      current_start = current_end + 1;
    }
  }

  // Signal that all initial tasks have been submitted
  // Use release semantics to ensure previous writes (task pushes, counter
  // increments) are visible before the flag is set.
  g_all_tasks_submitted_ws.store(true, std::memory_order_release);

  // Wait for all worker threads to complete
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join(); // Wait for all threads to finish.
    }
  }

  return true;
}
