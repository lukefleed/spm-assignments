#include "dynamic_scheduler.h"
#include "collatz.h" // For collatz_steps and find_max_steps_in_subrange
#include <atomic>    // For std::atomic, memory orders
#include <iostream>  // For verbose/error output
#include <limits>    // For std::numeric_limits
#include <numeric>   // Potentially useful, though not directly used here
#include <optional>  // For std::optional return types
#include <random>    // Could be used for randomized stealing (not currently)
#include <thread>    // For std::thread, std::this_thread::yield
#include <vector>    // For std::vector

//----------------------------------------------------------------------------
// TaskQueue (Simple Centralized Queue Implementation)
//----------------------------------------------------------------------------

/**
 * @brief Pushes a task onto the centralized task queue.
 *        This method is thread-safe.
 * @param task The Task object to be added.
 * @note Notifies one waiting worker thread after adding the task. If the queue
 *       is closed, the task is discarded.
 */
void TaskQueue::push(Task task) {
  {
    // Lock the mutex to ensure exclusive access to the queue.
    std::lock_guard<std::mutex> lock(mutex_);
    // If the queue has been closed (signaling no more tasks will be added),
    // simply return without adding the task.
    if (closed_) {
      return;
    }
    // Move the task into the queue.
    queue_.push(std::move(task));
  } // Mutex is released here by lock_guard destructor.

  // Notify *one* waiting thread. This is efficient as typically only one
  // thread needs to wake up to process the newly added task.
  cond_var_.notify_one();
}

/**
 * @brief Pops a task from the centralized task queue.
 *        If the queue is empty, this method blocks until a task becomes
 * available or the queue is closed. This method is thread-safe.
 * @return An std::optional<Task> containing the task if successful.
 *         std::nullopt if the queue is closed and empty.
 */
std::optional<Task> TaskQueue::pop() {
  // Acquire the lock. unique_lock is needed for condition variable waiting.
  std::unique_lock<std::mutex> lock(mutex_);

  // Wait using the condition variable. The thread blocks *atomically*
  // (releasing the lock) until the predicate `[this] { return !queue_.empty()
  // || closed_; }` becomes true AND the condition variable is notified. The
  // lock is re-acquired before the wait returns. This prevents spurious wakeups
  // from causing issues and ensures the thread only proceeds if there's work or
  // the queue is shutting down.
  cond_var_.wait(lock, [this] { return !queue_.empty() || closed_; });

  // After waking up and re-acquiring the lock:
  // If the queue is empty, it must be because it was closed. Return nullopt
  // to signal the worker thread to terminate.
  if (queue_.empty()) {
    return std::nullopt;
  }

  // Otherwise, a task is available. Move it out of the queue.
  Task task = std::move(queue_.front());
  queue_.pop();
  return task;
}

/**
 * @brief Closes the task queue, signaling that no more tasks will be added.
 *        This method is thread-safe.
 * @note Wakes up all waiting worker threads so they can check the closed status
 *       and terminate if the queue is also empty.
 */
void TaskQueue::close() {
  {
    // Lock the mutex to safely modify the closed_ flag.
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  } // Mutex is released here.

  // Notify *all* waiting threads. This is crucial to ensure that all blocked
  // workers wake up, check the `closed_` flag, and exit if the queue is empty.
  cond_var_.notify_all();
}

//----------------------------------------------------------------------------
// WorkStealingQueue (Per-Thread Deque Implementation)
//----------------------------------------------------------------------------

/**
 * @brief Pushes a task onto the back (LIFO end) of this thread's local queue.
 *        This method is intended to be called primarily by the owner thread.
 *        Thread-safe due to the internal mutex.
 * @param task The Task object to be added.
 */
void WorkStealingQueue::push(Task task) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Using push_back maintains the LIFO (Last-In, First-Out) order for the owner
  // thread's pop().
  queue_.push_back(std::move(task));
}

/**
 * @brief Pops a task from the back (LIFO end) of this thread's local queue.
 *        This method is intended to be called primarily by the owner thread.
 *        Thread-safe due to the internal mutex.
 * @return An std::optional<Task> containing the task if the queue is not empty,
 *         std::nullopt otherwise.
 */
std::optional<Task> WorkStealingQueue::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  // Using pop_back adheres to the LIFO principle for the owner thread.
  // This often improves cache locality, as the thread is likely to work
  // on recently added (and potentially related) tasks.
  Task task = std::move(queue_.back());
  queue_.pop_back();
  return task;
}

/**
 * @brief Steals a task from the front (FIFO end) of this queue.
 *        This method is intended to be called by *other* threads (thieves).
 *        Thread-safe due to the internal mutex.
 * @return An std::optional<Task> containing the stolen task if the queue is not
 * empty, std::nullopt otherwise.
 */
std::optional<Task> WorkStealingQueue::steal() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  // Thieves steal from the front (pop_front), adhering to FIFO for stolen
  // tasks. This reduces contention with the owner thread, which primarily
  // accesses the back. It also tends to steal older tasks, potentially larger
  // ones if tasks were generated hierarchically.
  Task task = std::move(queue_.front());
  queue_.pop_front();
  return task;
}

/**
 * @brief Checks if the queue is empty. Thread-safe.
 * @return True if the queue contains no tasks, false otherwise.
 */
bool WorkStealingQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

/**
 * @brief Returns the current number of tasks in the queue. Thread-safe.
 * @return The number of tasks currently present.
 */
size_t WorkStealingQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

//----------------------------------------------------------------------------
// Global State for Work Stealing Scheduler
//----------------------------------------------------------------------------

/**
 * @brief Global atomic counter tracking the number of tasks currently being
 * processed or waiting in any queue within the work-stealing system. Used for
 * termination detection.
 */
std::atomic<size_t> g_pending_tasks_ws{0};

/**
 * @brief Global atomic flag indicating whether the main thread has finished
 * submitting all initial tasks to the work-stealing queues. Used for
 * termination detection.
 */
std::atomic<bool> g_all_tasks_submitted_ws{false};

//----------------------------------------------------------------------------
// Dynamic Scheduler Worker Functions
//----------------------------------------------------------------------------

/**
 * @brief Worker function for the simple dynamic scheduler (using TaskQueue).
 *
 * Continuously attempts to pop tasks from the shared `TaskQueue`. If a task is
 * obtained, it processes the assigned sub-range and atomically updates the
 * maximum step count for the corresponding original range. The loop terminates
 * when `queue.pop()` returns `std::nullopt`, indicating the queue is closed and
 * empty.
 *
 * @param thread_id The ID of this worker thread (currently unused, but good
 * practice to pass).
 * @param queue The shared `TaskQueue` instance.
 * @param results_out The shared vector for storing results, accessed
 * atomically.
 *
 * @note Uses `compare_exchange_weak` in a loop for atomic maximum update.
 * `relaxed` memory order is sufficient because the update is self-contained
 * (doesn't need to synchronize other variables) and the final result
 * consistency is ensured by joining the threads later. The yield helps reduce
 * contention during the CAS loop on highly contested updates, though it adds
 * overhead.
 */
void dynamic_worker(int thread_id [[maybe_unused]], TaskQueue &queue,
                    std::vector<RangeResult> &results_out) {
  while (true) {
    // Block until a task is available or the queue is closed & empty.
    std::optional<Task> task_opt = queue.pop();

    // If pop returns nullopt, the queue is closed and empty, so terminate.
    if (!task_opt) {
      break;
    }

    Task task = *task_opt; // Extract the task.

    // Perform the actual computation for the task's sub-range.
    ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

    // Atomically update the result for the original range this task belongs to.
    if (local_max_steps > 0) { // Avoid unnecessary atomics if max is 0.
      ull current_max = results_out[task.original_range_index].max_steps.load(
          std::memory_order_relaxed);
      // Loop using compare_exchange_weak until update succeeds or local_max is
      // no longer greater.
      while (local_max_steps > current_max) {
        if (results_out[task.original_range_index]
                .max_steps.compare_exchange_weak(
                    current_max, local_max_steps,
                    std::memory_order_release, // Use release on success for
                                               // potential visibility gains.
                    std::memory_order_relaxed // Relaxed on failure is standard.
                    )) {
          break; // Update successful.
        }
        // CAS failed, current_max was updated by the call. Loop continues.
        // Consider yielding briefly if contention is extremely high, though
        // often unnecessary. std::this_thread::yield();
      }
    }
  } // End while(true)
}

/**
 * @brief Worker function for the work-stealing dynamic scheduler.
 *
 * Implements the core work-stealing logic:
 * 1. Prioritize popping tasks from its own local queue (LIFO).
 * 2. If the local queue is empty, attempt to steal tasks from other threads'
 * queues (FIFO). A simple round-robin victim selection is used. A basic backoff
 * mechanism is implemented to reduce contention when repeatedly failing to
 * steal from a victim.
 * 3. If a task is obtained (either popped or stolen), process it and atomically
 * update results. Crucially, decrement the global pending task counter *after*
 * processing.
 * 4. If no task is found locally or stolen, check for the termination
 * condition:
 *    - Have all initial tasks been submitted? (`g_all_tasks_submitted_ws`)
 *    - Is the global pending task count zero? (`g_pending_tasks_ws`)
 * 5. If the termination condition seems met, perform a stricter check and
 * potentially exit.
 * 6. If no work is found and not terminating, yield the thread to avoid
 * busy-waiting.
 *
 * @param thread_id The ID of this worker thread.
 * @param num_threads The total number of worker threads.
 * @param queues A reference to the vector containing all `WorkStealingQueue`
 * instances.
 * @param results_out The shared vector for storing results, accessed
 * atomically.
 *
 * @note Termination detection in distributed/work-stealing systems is
 * non-trivial. This implementation relies on a global atomic counter
 * (`g_pending_tasks_ws`) and a flag
 *       (`g_all_tasks_submitted_ws`). A task is only decremented *after* it's
 * fully processed. `acquire` memory order is used when reading these atomics in
 * the termination check to ensure visibility of prior `release` operations
 * (task decrements and setting the submitted flag). A `seq_cst` fence is added
 * for a stronger (though potentially overkill) guarantee before the final
 * check, aiming to prevent race conditions where a thread might exit
 * prematurely while another thread is still processing or about to steal its
 * last task.
 */
void dynamic_work_stealing_worker(
    int thread_id, int num_threads,
    std::vector<WorkStealingQueue>
        &queues, // Needs to be non-const for stealing
    std::vector<RangeResult> &results_out) {

  // Simple per-thread state for the backoff mechanism. Could be more
  // sophisticated.
  thread_local std::vector<int> failed_attempts(num_threads, 0);
  thread_local std::vector<int> backoff_countdown(num_threads, 0);
  const int MAX_FAILED_ATTEMPTS = 3; // Threshold to trigger backoff.
  const int BACKOFF_MULTIPLIER = 5;  // Determines initial backoff duration.

  while (true) {
    std::optional<Task> task_opt;

    // 1. Try to pop from the thread's own queue (LIFO).
    task_opt = queues[thread_id].pop();

    // 2. If the local queue was empty, attempt to steal from others.
    if (!task_opt) {
      // Simple round-robin victim selection starting from the next thread.
      int victim_id = (thread_id + 1) % num_threads;
      for (int i = 0; i < num_threads - 1;
           ++i) { // Try stealing from n-1 other queues
        // Don't steal from self. Check if the potential victim is currently in
        // backoff for this thread.
        if (victim_id != thread_id && backoff_countdown[victim_id] == 0) {
          task_opt = queues[victim_id].steal(); // Attempt FIFO steal.
          if (task_opt) {
            // Steal succeeded! Reset failed attempts counter for this victim.
            failed_attempts[victim_id] = 0;
            break; // Got a task, stop searching.
          } else {
            // Steal failed. Increment failed attempts.
            failed_attempts[victim_id]++;
            // If failed attempts exceed threshold, initiate backoff.
            if (failed_attempts[victim_id] > MAX_FAILED_ATTEMPTS) {
              backoff_countdown[victim_id] =
                  failed_attempts[victim_id] * BACKOFF_MULTIPLIER;
              // Could cap the backoff time if needed.
            }
          }
        }
        // Advance to the next potential victim.
        victim_id = (victim_id + 1) % num_threads;
      } // End stealing loop

      // Decrease backoff counters for all potential victims after a steal
      // round.
      for (int j = 0; j < num_threads; ++j) {
        if (backoff_countdown[j] > 0) {
          backoff_countdown[j]--;
        }
      }
    } // End stealing attempt block

    // 3. Process the task if one was obtained (popped or stolen).
    if (task_opt) {
      Task task = *task_opt; // Extract the task.

      // Perform the computation.
      ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

      // Atomically update the result.
      if (local_max_steps > 0) {
        ull current_max = results_out[task.original_range_index].max_steps.load(
            std::memory_order_relaxed);
        while (local_max_steps > current_max) {
          if (results_out[task.original_range_index]
                  .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed)) {
            break; // Update successful.
          }
        }
      }

      // Crucially, decrement the global pending task counter *after*
      // processing. Use release semantics to make the decrement visible to
      // other threads checking the termination condition.
      g_pending_tasks_ws.fetch_sub(1, std::memory_order_release);

    } else {
      // 4. No work found locally or via stealing. Check for termination.
      // Load the submission flag with acquire semantics to synchronize with the
      // main thread's release-store. Load the pending task counter with acquire
      // semantics to synchronize with worker threads' release-fetch_sub.
      if (g_all_tasks_submitted_ws.load(std::memory_order_acquire) &&
          g_pending_tasks_ws.load(std::memory_order_acquire) == 0) {

        // Potential termination condition met.
        // Double-checking can help prevent rare race conditions where a task
        // is in-flight between being stolen and being processed, or where the
        // counter briefly hits zero before the last task is fully processed and
        // decremented.

        // Optional: Add a check across *all* queues for emptiness for extra
        // robustness. bool all_queues_really_empty = true; for(int q_idx = 0;
        // q_idx < num_threads; ++q_idx) {
        //     if (!queues[q_idx].empty()) {
        //         all_queues_really_empty = false;
        //         break;
        //     }
        // }
        // if (!all_queues_really_empty) continue; // False alarm, continue
        // working.

        // Stronger check: Re-check pending tasks after ensuring memory
        // visibility. A seq_cst fence enforces maximum visibility across all
        // threads, making it highly likely that if the counter is *still* zero
        // after the fence, all processing is truly done. This is conservative
        // but safer.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (g_pending_tasks_ws.load(std::memory_order_relaxed) ==
            0) { // Relaxed is okay for the re-check itself
          break; // Confirmed termination condition. Exit the worker loop.
        }
        // If counter wasn't zero on re-check, it was a transient state, so
        // continue.
      }

      // 5. If not terminating and no work was found, yield the CPU.
      // This prevents the idle thread from consuming 100% CPU in a tight loop.
      std::this_thread::yield();
    }
  } // End while(true)
}

//----------------------------------------------------------------------------
// Dynamic Scheduler Execution Functions
//----------------------------------------------------------------------------

/**
 * @brief Executes the Collatz computation using the simple dynamic task queue
 * scheduler.
 *
 * Creates a single shared task queue, spawns worker threads that pull tasks
 * from it, and feeds tasks (chunks of ranges) into the queue. Waits for all
 * threads to finish.
 *
 * @param config Configuration parameters (num_threads, chunk_size, ranges).
 * @param results_out Vector to store the final results.
 * @return True if execution setup and completion were successful, false on
 * invalid config.
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out) {
  // Validate essential configuration parameters.
  if (config.num_threads <= 0 || config.chunk_size == 0) {
    std::cerr << "Error: Dynamic task queue requires positive num_threads and "
                 "chunk_size."
              << std::endl;
    return false;
  }

  TaskQueue task_queue; // Centralized queue.
  std::vector<std::thread> threads;

  // Initialize results vector, clearing previous content.
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r); // Initialize with original range, max_steps=0.
  }

  // Launch worker threads.
  threads.reserve(config.num_threads);
  for (int i = 0; i < config.num_threads; ++i) {
    // Pass queue and results by reference using std::ref.
    threads.emplace_back(dynamic_worker, i, std::ref(task_queue),
                         std::ref(results_out));
  }

  // Main thread acts as the producer, dividing ranges into tasks (chunks).
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue; // Skip invalid ranges.

    ull current_start = range.start;
    while (current_start <= range.end) {
      // Calculate the end of the current chunk, carefully handling potential
      // ull overflow.
      ull current_chunk_end =
          (current_start >
           std::numeric_limits<ull>::max() - (config.chunk_size - 1))
              ? range.end // Avoid overflow, clamp to range end.
              : current_start + config.chunk_size - 1;

      // Ensure the chunk end doesn't exceed the actual range end.
      ull current_end = std::min(range.end, current_chunk_end);

      // Push the task definition onto the queue.
      task_queue.push({current_start, current_end,
                       i}); // {task_start, task_end, original_range_index}

      // Check if this was the last chunk of the range.
      if (current_end == range.end)
        break;

      // Advance start for the next chunk.
      // Check for overflow before incrementing.
      if (current_end == std::numeric_limits<ull>::max())
        break; // Cannot advance further.
      current_start = current_end + 1;
    }
  }

  // All tasks have been generated and pushed. Close the queue to signal
  // workers.
  task_queue.close();

  // Wait for all worker threads to complete their execution and exit.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  return true; // Indicate successful completion.
}

/**
 * @brief Executes the Collatz computation using the dynamic work-stealing
 * scheduler.
 *
 * Creates per-thread work-stealing queues, spawns worker threads that
 * prioritize their own queue but steal from others when idle. The main thread
 * distributes initial tasks across the queues. Relies on global atomics for
 * termination detection.
 *
 * @param config Configuration parameters (num_threads, chunk_size, ranges).
 * @param results_out Vector to store the final results.
 * @return True if execution setup and completion were successful, false on
 * invalid config.
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out) {
  // Validate essential configuration parameters.
  if (config.num_threads <= 0 || config.chunk_size == 0) {
    std::cerr << "Error: Dynamic work stealing requires positive num_threads "
                 "and chunk_size."
              << std::endl;
    return false;
  }

  // Reset global atomic state for this run. Relaxed is sufficient for
  // initialization.
  g_pending_tasks_ws.store(0, std::memory_order_relaxed);
  g_all_tasks_submitted_ws.store(false, std::memory_order_relaxed);

  // Create the vector of per-thread queues.
  std::vector<WorkStealingQueue> queues(config.num_threads);
  std::vector<std::thread> threads;

  // Initialize results vector.
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r);
  }

  // Launch worker threads.
  threads.reserve(config.num_threads);
  for (int i = 0; i < config.num_threads; ++i) {
    // Pass the vector of queues and results by reference.
    threads.emplace_back(dynamic_work_stealing_worker, i, config.num_threads,
                         std::ref(queues), std::ref(results_out));
  }

  // Main thread acts as the initial task distributor.
  size_t current_queue_idx = 0; // Index for round-robin distribution.
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue; // Skip invalid ranges.

    ull current_start = range.start;
    while (current_start <= range.end) {
      // Calculate chunk boundaries, handling overflow.
      ull current_chunk_end = (current_start > std::numeric_limits<ull>::max() -
                                                   (config.chunk_size - 1))
                                  ? range.end
                                  : current_start + config.chunk_size - 1;
      ull current_end = std::min(range.end, current_chunk_end);

      // Increment pending task count *before* pushing the task.
      // Relaxed order is acceptable here; the synchronization that matters is
      // the release-decrement after processing and the acquire-read during
      // termination check.
      g_pending_tasks_ws.fetch_add(1, std::memory_order_relaxed);

      // Distribute the task to a worker queue using round-robin.
      queues[current_queue_idx].push({current_start, current_end, i});
      current_queue_idx = (current_queue_idx + 1) % config.num_threads;

      if (current_end == range.end)
        break; // Last chunk of this range.
      if (current_end == std::numeric_limits<ull>::max())
        break; // Cannot advance further.
      current_start = current_end + 1;
    }
  }

  // Signal that the main thread has finished submitting all initial tasks.
  // Use release semantics to ensure that all prior writes (task pushes, counter
  // increments) become visible to worker threads *before* they see this flag
  // set to true. This is crucial for the termination condition logic in the
  // workers.
  g_all_tasks_submitted_ws.store(true, std::memory_order_release);

  // Wait for all worker threads to complete.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Sanity check: After joining, the pending task counter should ideally be
  // zero. If not, it might indicate a logic error in task counting or
  // termination.
  if (config.verbose &&
      g_pending_tasks_ws.load(std::memory_order_relaxed) != 0) {
    std::cerr << "Warning: Pending tasks counter is non-zero ("
              << g_pending_tasks_ws.load(std::memory_order_relaxed)
              << ") after joining threads." << std::endl;
  }

  return true; // Indicate successful completion.
}
