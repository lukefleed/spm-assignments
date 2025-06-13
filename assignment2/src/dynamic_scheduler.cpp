#include "dynamic_scheduler.h"
#include "collatz.h" // For find_max_steps_in_subrange
#include <atomic>    // For std::atomic, memory orders
#include <iostream>  // For verbose/error output
#include <limits>    // For std::numeric_limits
#include <optional>  // For std::optional return types
#include <thread>    // For std::thread, std::this_thread::yield
#include <vector>    // For std::vector

//----------------------------------------------------------------------------
// TaskQueue (Simple Centralized Queue Implementation)
//----------------------------------------------------------------------------

void TaskQueue::push(Task task) {
  {
    // Lock the mutex to ensure exclusive access to the queue.
    std::lock_guard<std::mutex> lock(mutex_);
    // If the queue has been closed (signaling no more tasks will be added),
    // simply return without adding the task.
    if (closed_) {
      return;
    }
    // Move the task into the queue. Using move is efficient if Task has
    // move semantics.
    queue_.push(std::move(task));
  } // Mutex is released here by lock_guard destructor.

  // Notify *one* waiting thread. This is efficient as typically only one
  // thread needs to wake up to process the newly added task. Other waiting
  // threads remain asleep until notified again.
  cond_var_.notify_one();
}

std::optional<Task> TaskQueue::pop() {
  // Acquire the lock. unique_lock is needed for condition variable waiting.
  std::unique_lock<std::mutex> lock(mutex_);

  // Wait using the condition variable. The thread blocks *atomically*
  // (releasing the lock) until the predicate `[this] { return !queue_.empty()
  // || closed_; }` becomes true AND the condition variable is notified.
  // The lock is re-acquired before the wait returns. This prevents spurious
  // wakeups from causing issues and ensures the thread only proceeds if
  // there's work or the queue is shutting down.
  cond_var_.wait(lock, [this] { return !queue_.empty() || closed_; });

  // After waking up and re-acquiring the lock:
  // If the queue is empty, it must be because it was closed (due to the
  // predicate). Return nullopt to signal the worker thread to terminate.
  if (queue_.empty()) {
    return std::nullopt;
  }

  // Otherwise, a task is available. Move it out of the queue.
  // std::move is technically redundant for queue.front() followed by pop(),
  // but it's explicit about the intent.
  Task task = std::move(queue_.front());
  queue_.pop();
  return task; // Return the task wrapped in std::optional.
}

void TaskQueue::close() {
  {
    // Lock the mutex to safely modify the closed_ flag.
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  } // Mutex is released here.

  // Notify *all* waiting threads. This is crucial to ensure that all blocked
  // workers wake up, check the `closed_` flag in their `pop()` predicate,
  // and exit if the queue is also empty.
  cond_var_.notify_all();
}

//----------------------------------------------------------------------------
// WorkStealingQueue (Mutex-Based Per-Thread Deque Implementation)
//----------------------------------------------------------------------------

void WorkStealingQueue::push(Task task) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Using push_back maintains the LIFO (Last-In, First-Out) order for the owner
  // thread's pop(). This typically improves cache locality as the owner works
  // on the most recently added tasks.
  queue_.push_back(std::move(task));
}

std::optional<Task> WorkStealingQueue::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  // Using pop_back adheres to the LIFO principle for the owner thread.
  Task task = std::move(queue_.back());
  queue_.pop_back();
  return task;
}

std::optional<Task> WorkStealingQueue::steal() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  // Thieves steal from the front (pop_front), adhering to FIFO for stolen
  // tasks. This reduces contention with the owner thread (which accesses the
  // back) and tends to steal older, potentially larger "chunks" of work if
  // tasks were generated hierarchically.
  Task task = std::move(queue_.front());
  queue_.pop_front();
  return task;
}

bool WorkStealingQueue::empty() const {
  // Lock is required even for checking empty, as another thread might be
  // modifying the queue concurrently.
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

size_t WorkStealingQueue::size() const {
  // Lock is required for thread-safe size check.
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

//----------------------------------------------------------------------------
// Global State for Work Stealing Scheduler
//----------------------------------------------------------------------------

/**
 * @brief Global atomic counter tracking the number of tasks currently
 * submitted to the system but not yet fully processed by a worker. Used for
 * termination detection in the work-stealing scheduler. Incremented before
 * pushing, decremented after processing.
 */
std::atomic<size_t> g_pending_tasks_ws{0};

/**
 * @brief Global atomic flag indicating whether the main thread (or task
 * submitter) has finished submitting all initial tasks to the work-stealing
 * queues. Used in conjunction with g_pending_tasks_ws for termination
 * detection.
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
 * @param thread_id The logical ID of this worker thread (0 to num_threads-1).
 * @param queue Reference to the shared `TaskQueue` instance.
 * @param results_out Reference to the shared vector for storing results,
 * accessed atomically.
 *
 * @note Uses `compare_exchange_weak` in a loop for atomic maximum update.
 * `relaxed` memory order is sufficient for reading the current maximum because
 * the update logic itself ensures correctness. `release` is used on successful
 * write to potentially make the update visible slightly sooner to other
 * threads, although the final synchronization happens at thread join.
 */
void dynamic_worker(int thread_id [[maybe_unused]], TaskQueue &queue,
                    std::vector<RangeResult> &results_out) {
  while (true) {
    // Block until a task is available or the queue is closed & empty.
    std::optional<Task> task_opt = queue.pop();

    // If pop returns nullopt, the queue is closed and empty, so terminate.
    if (!task_opt) {
      break; // Exit the worker loop.
    }

    Task task = *task_opt; // Extract the task from optional.

    // Perform the actual computation for the task's sub-range.
    ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

    // Atomically update the result for the original range this task belongs to.
    if (local_max_steps > 0) { // Avoid unnecessary atomics if max is 0.
      // Get a reference to the atomic member for clarity.
      std::atomic<ull> &target_max =
          results_out[task.original_range_index].max_steps;

      // Load the current maximum stored for this range. Relaxed is fine as we
      // will CAS.
      ull current_max = target_max.load(std::memory_order_relaxed);

      // Loop using compare_exchange_weak: update only if our local_max_steps is
      // still greater than the current maximum stored in the shared results.
      while (local_max_steps > current_max) {
        // Attempt to replace current_max with local_max_steps.
        // - On success: `release` ensures prior writes (like the computation)
        //               are visible before this store.
        // - On failure: `relaxed` is sufficient; current_max gets updated with
        //               the value that caused the failure.
        if (target_max.compare_exchange_weak(current_max, local_max_steps,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
          break; // Update successful, exit the CAS loop.
        }
        // CAS failed: current_max now holds the newer value read from memory.
        // The loop condition (local_max_steps > current_max) will re-evaluate.
      }
    }
  } // End while(true)
}

/**
 * @brief Worker function for the dynamic work-stealing scheduler.
 *
 * Implements the core work-stealing logic:
 * 1. Prioritize popping tasks from its own local queue (`queues[thread_id]`,
 * LIFO).
 * 2. If the local queue is empty, attempt to steal tasks from other threads'
 *    queues (FIFO, using `steal()`). A simple round-robin victim selection is
 *    used.
 * 3. Implements a basic exponential backoff mechanism for stealing attempts to
 *    reduce contention when queues are frequently empty.
 * 4. If a task is obtained (either popped or stolen), process it and atomically
 *    update the corresponding result in `results_out`.
 * 5. Crucially, decrement the global pending task counter
 * (`g_pending_tasks_ws`) *after* processing is complete, using `release`
 * semantics.
 * 6. If no task is found locally or stolen, check for the global termination
 *    condition using `g_all_tasks_submitted_ws` and `g_pending_tasks_ws` with
 *    `acquire` semantics.
 * 7. Performs a final consistency check with a `seq_cst` fence before exiting
 *    to minimize the chance of premature termination.
 * 8. If no work is found and not terminating, `yield` the thread to avoid
 *    busy-waiting and consuming excessive CPU.
 *
 * @param thread_id The logical ID of this worker thread (0 to num_threads-1).
 * @param num_threads The total number of worker threads in the system.
 * @param queues A reference to the vector containing all `WorkStealingQueue`
 * instances (one per thread). Must be non-const to allow stealing.
 * @param results_out Reference to the shared vector for storing results,
 * accessed atomically.
 *
 * @note Termination detection in distributed work-stealing systems is
 * non-trivial. This implementation relies on the combination of the task
 * counter and the submission flag, along with appropriate memory ordering, to
 * ensure all tasks are processed before threads terminate. The `seq_cst` fence
 * provides a strong guarantee against reordering around the final check.
 */
void dynamic_work_stealing_worker(int thread_id, int num_threads,
                                  std::vector<WorkStealingQueue> &queues,
                                  std::vector<RangeResult> &results_out) {

  // Per-thread state for the stealing backoff mechanism.
  // `thread_local` ensures each thread has its own independent copies.
  thread_local std::vector<int> failed_steal_attempts(num_threads, 0);
  thread_local std::vector<int> backoff_countdown(num_threads, 0);
  const int MAX_FAILED_STEAL_ATTEMPTS =
      3; // Consecutive failures before backing off.
  const int BACKOFF_MULTIPLIER =
      5; // Factor to determine initial backoff duration (in yields/loops).

  while (true) {
    std::optional<Task> task_opt;

    // 1. Try to pop from the thread's own queue (LIFO).
    task_opt = queues[thread_id].pop();

    // 2. If the local queue was empty, attempt to steal from others.
    if (!task_opt) {
      // Simple round-robin victim selection, starting from the next thread.
      int victim_id = (thread_id + 1) % num_threads;
      for (int i = 0; i < num_threads - 1;
           ++i) { // Try stealing from all *other* queues.

        // Skip self and skip victims currently in backoff for this thread.
        if (victim_id != thread_id && backoff_countdown[victim_id] == 0) {
          task_opt = queues[victim_id].steal(); // Attempt FIFO steal.

          if (task_opt) {
            // Steal succeeded! Reset failed attempts counter for this victim.
            failed_steal_attempts[victim_id] = 0;
            break; // Got a task, stop searching for victims.
          } else {
            // Steal failed for this victim. Increment failed attempts.
            failed_steal_attempts[victim_id]++;
            // If failed attempts exceed threshold, initiate backoff.
            if (failed_steal_attempts[victim_id] > MAX_FAILED_STEAL_ATTEMPTS) {
              // Calculate backoff duration (can be capped if needed).
              backoff_countdown[victim_id] =
                  failed_steal_attempts[victim_id] * BACKOFF_MULTIPLIER;
            }
          }
        }
        // Advance to the next potential victim (wraps around using modulo).
        victim_id = (victim_id + 1) % num_threads;
      } // End stealing loop

      // After attempting to steal (or skipping due to backoff), decrease active
      // backoff counters.
      for (int j = 0; j < num_threads; ++j) {
        if (backoff_countdown[j] > 0) {
          backoff_countdown[j]--;
        }
      }
    } // End stealing attempt block

    // 3. Process the task if one was obtained (either popped or stolen).
    if (task_opt) {
      Task task = *task_opt; // Extract the task from optional.

      // Perform the actual computation.
      ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

      // Atomically update the result (same logic as in dynamic_worker).
      if (local_max_steps > 0) {
        std::atomic<ull> &target_max =
            results_out[task.original_range_index].max_steps;
        ull current_max = target_max.load(std::memory_order_relaxed);
        while (local_max_steps > current_max) {
          if (target_max.compare_exchange_weak(current_max, local_max_steps,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
            break;
          }
        }
      }

      // 5. Crucially, decrement the global pending task counter *after*
      // processing is complete. Use release semantics to ensure that the task
      // processing and result update are visible before the counter change,
      // synchronizing with the acquire loads in the termination check.
      g_pending_tasks_ws.fetch_sub(1, std::memory_order_release);

    } else {
      // 4. No work found locally or via stealing. Check for termination.

      // Load the submission flag with acquire semantics. This synchronizes with
      // the main thread's release-store, ensuring we see the correct flag value
      // and any task pushes that happened before it.
      bool all_submitted =
          g_all_tasks_submitted_ws.load(std::memory_order_acquire);

      // Load the pending task counter with acquire semantics. This synchronizes
      // with the release-fetch_sub operations from other workers, ensuring we
      // see the effects of their completed tasks.
      size_t pending_tasks = g_pending_tasks_ws.load(std::memory_order_acquire);

      if (all_submitted && pending_tasks == 0) {
        // Potential termination condition met. Need a stronger check to avoid
        // races where a task is in-flight (stolen but not yet processed/counted
        // down) or the counter momentarily hits zero.

        // A full sequential consistency fence ensures that all memory
        // operations preceding the fence in this thread are visible globally
        // before any operations following the fence, and vice-versa for other
        // threads. This drastically reduces the window for race conditions in
        // termination detection.
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Re-check the pending task counter *after* the fence. If it's still
        // zero, it's highly likely that all work is truly done. Relaxed order
        // is sufficient for this second read as the fence provided the primary
        // synchronization.
        if (g_pending_tasks_ws.load(std::memory_order_relaxed) == 0) {
          break; // Confirmed termination condition. Exit the worker loop.
        }
        // If counter wasn't zero on re-check, it was a transient state or a
        // task completed between the first check and the fence. Continue
        // working/stealing.
      }

      // 6. If not terminating and no work was found, yield the CPU.
      // This prevents the idle thread from consuming 100% CPU in a tight loop
      // while waiting for work or the termination condition.
      std::this_thread::yield();
    }
  } // End while(true)
}

//----------------------------------------------------------------------------
// Dynamic Scheduler Execution Functions
//----------------------------------------------------------------------------

bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out) {
  // Validate essential configuration parameters.
  if (config.num_threads <= 0 || config.chunk_size == 0) {
    std::cerr << "Error: Dynamic task queue requires positive num_threads ("
              << config.num_threads << ") and chunk_size (" << config.chunk_size
              << ")." << std::endl;
    return false;
  }

  TaskQueue task_queue; // The single, centralized queue.
  std::vector<std::thread> threads;

  // Initialize results vector, clearing previous content and reserving space.
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (const auto &r : config.ranges) {
    // Initialize with original range info, max_steps will be updated
    // atomically.
    results_out.emplace_back(r);
  }

  // Launch worker threads.
  threads.reserve(config.num_threads);
  for (unsigned int i = 0; i < config.num_threads; ++i) {
    // Pass queue and results by reference using std::ref.
    // Pass the thread ID 'i'.
    threads.emplace_back(dynamic_worker, i, std::ref(task_queue),
                         std::ref(results_out));
  }

  // Main thread acts as the producer, dividing ranges into tasks (chunks).
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end) {
      if (config.verbose) {
        std::cerr << "Warning: Skipping invalid range [" << range.start << ", "
                  << range.end << "]" << std::endl;
      }
      continue; // Skip invalid ranges.
    }

    ull current_start = range.start;
    while (current_start <= range.end) {
      // Calculate the end of the current chunk, carefully handling potential
      // ull overflow when adding chunk_size.
      ull current_chunk_end = 0;
      // Check for overflow before adding chunk_size - 1
      if (current_start >
          std::numeric_limits<ull>::max() - (config.chunk_size - 1)) {
        current_chunk_end = std::numeric_limits<ull>::max(); // Clamp to max ull
      } else {
        current_chunk_end = current_start + config.chunk_size - 1;
      }

      // Ensure the chunk end doesn't exceed the actual range end.
      ull current_end = std::min(range.end, current_chunk_end);

      // Create and push the task definition onto the queue.
      task_queue.push({current_start, current_end,
                       i}); // {task_start, task_end, original_range_index}

      // Check if this was the last chunk of the range.
      if (current_end == range.end) {
        break;
      }

      // Advance start for the next chunk. Check for overflow before
      // incrementing.
      if (current_end == std::numeric_limits<ull>::max()) {
        break; // Cannot advance further if we reached max ull.
      }
      current_start = current_end + 1;
    }
  }

  // All tasks have been generated and pushed. Close the queue to signal workers
  // that no more tasks will arrive.
  task_queue.close();

  // Wait for all worker threads to complete their execution and exit.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  return true; // Indicate successful completion.
}

bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out) {
  // Validate essential configuration parameters.
  if (config.num_threads <= 0 || config.chunk_size == 0) {
    std::cerr << "Error: Dynamic work stealing requires positive num_threads ("
              << config.num_threads << ") and chunk_size (" << config.chunk_size
              << ")." << std::endl;
    return false;
  }

  // Reset global atomic state for this run. Relaxed is sufficient for
  // initialization as no other threads are accessing them yet.
  g_pending_tasks_ws.store(0, std::memory_order_relaxed);
  g_all_tasks_submitted_ws.store(false, std::memory_order_relaxed);

  // Create the vector of per-thread work-stealing queues.
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
  for (unsigned int i = 0; i < config.num_threads; ++i) {
    // Pass the vector of queues and results by reference. Pass thread ID 'i'
    // and total thread count.
    threads.emplace_back(dynamic_work_stealing_worker, i, config.num_threads,
                         std::ref(queues), std::ref(results_out));
  }

  // Main thread acts as the initial task distributor.
  size_t current_queue_idx = 0; // Index for round-robin distribution.
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end) {
      if (config.verbose) {
        std::cerr << "Warning: Skipping invalid range [" << range.start << ", "
                  << range.end << "]" << std::endl;
      }
      continue; // Skip invalid ranges.
    }

    ull current_start = range.start;
    while (current_start <= range.end) {
      // Calculate chunk boundaries, handling overflow.
      ull current_chunk_end = 0;
      if (current_start >
          std::numeric_limits<ull>::max() - (config.chunk_size - 1)) {
        current_chunk_end = std::numeric_limits<ull>::max();
      } else {
        current_chunk_end = current_start + config.chunk_size - 1;
      }
      ull current_end = std::min(range.end, current_chunk_end);

      // Increment pending task count *before* pushing the task.
      // Relaxed order is acceptable here; the main synchronization points are
      // the release-decrement after processing and the acquire-read during
      // termination check.
      g_pending_tasks_ws.fetch_add(1, std::memory_order_relaxed);

      // Distribute the task to a worker queue using round-robin.
      queues[current_queue_idx].push({current_start, current_end, i});
      current_queue_idx = (current_queue_idx + 1) % config.num_threads;

      if (current_end == range.end) {
        break; // Last chunk of this range.
      }
      if (current_end == std::numeric_limits<ull>::max()) {
        break; // Cannot advance further.
      }
      current_start = current_end + 1;
    }
  }

  // Signal that the main thread has finished submitting all initial tasks.
  // Use release semantics to ensure that all prior writes (task pushes to
  // queues, counter increments) become visible to worker threads *before* they
  // see this flag set to true. This is crucial for the termination condition
  // logic in the workers.
  g_all_tasks_submitted_ws.store(true, std::memory_order_release);

  // Wait for all worker threads to complete.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Sanity check: After joining all threads, the pending task counter should
  // ideally be zero. A non-zero value might indicate a logic error in task
  // counting, processing, or termination detection.
  size_t final_pending_tasks =
      g_pending_tasks_ws.load(std::memory_order_relaxed);
  if (config.verbose && final_pending_tasks != 0) {
    std::cerr << "Warning: Work-stealing pending tasks counter is non-zero ("
              << final_pending_tasks << ") after joining all threads."
              << std::endl;
  }

  return true; // Indicate successful completion.
}
