#ifndef DYNAMIC_SCHEDULER_H
#define DYNAMIC_SCHEDULER_H

#include "common_types.h" // Includes Task, RangeResult, Config, ull
#include <atomic>
#include <condition_variable> // For the simple TaskQueue if kept
#include <cstddef>            // For std::size_t, alignas
#include <deque>              // Added for TaskQueue implementation
#include <memory>             // For std::unique_ptr
#include <mutex>              // For the simple TaskQueue if kept
#include <optional>
#include <stdexcept> // For std::invalid_argument
#include <thread>
#include <vector>

// Forward declaration
class ChaseLevDeque;

//----------------------------------------------------------------------------
// TaskQueue (Simple Centralized Queue - Keep if needed for comparison/option)
//----------------------------------------------------------------------------

/**
 * @brief A simple thread-safe queue for centralized dynamic scheduling.
 *
 * Uses a standard mutex and condition variable for synchronization.
 * Suitable for Multiple Producer Multiple Consumer (MPMC) scenarios,
 * although performance may degrade under high contention compared to
 * more specialized concurrent queues.
 */
class TaskQueue {
private:
  std::deque<Task> queue_;   ///< Internal deque to store tasks.
  mutable std::mutex mutex_; ///< Mutex protecting access to the queue.
  std::condition_variable
      cond_var_;                    ///< Condition variable for waiting threads.
  std::atomic<bool> closed_{false}; ///< Flag indicating if the queue is closed.

public:
  /**
   * @brief Pushes a task onto the centralized task queue.
   * @param task The Task object to be added.
   * @note Notifies one waiting worker thread. If the queue is closed, the task
   * is discarded. Thread-safe.
   */
  void push(Task task);

  /**
   * @brief Pops a task from the centralized task queue.
   * @return std::optional<Task> containing the task if successful,
   *         std::nullopt if the queue is closed and empty. Blocks if empty
   * until task arrives or queue closes. Thread-safe.
   */
  std::optional<Task> pop();

  /**
   * @brief Closes the task queue, signaling no more tasks will be added.
   * @note Wakes up all waiting worker threads. Thread-safe.
   */
  void close();
};

//----------------------------------------------------------------------------
// ChaseLevDeque (Lock-Free Work-Stealing Deque)
//----------------------------------------------------------------------------

/**
 * @brief A work-stealing deque implementation based on the Chase-Lev algorithm.
 *
 * Designed for single-producer (owner) / multiple-consumer (thieves) scenarios.
 * The owner pushes and pops from the bottom (LIFO), while thieves steal from
 * the top (FIFO). Uses atomic operations for coordination, aiming for
 * lock-freedom on main paths.
 *
 * @note This implementation uses a fixed-size circular buffer. `push_bottom`
 * will fail if the deque is full.
 * @warning Assumes `Task` is efficiently movable and copy-assignable.
 * Correctness depends heavily on memory ordering semantics.
 */
class ChaseLevDeque {
private:
  // --- Member Order Corrected ---
  const std::size_t capacity_; ///< Capacity of the buffer (must be power of 2).
  std::unique_ptr<Task[]> buffer_; ///< Circular buffer storing tasks.

  // Align atomics to cache lines to reduce false sharing contention.
  alignas(64) std::atomic<std::size_t> top_; ///< Index for stealing
                                             ///< (incremented by stealers/pop).
  alignas(64)
      std::atomic<std::size_t> bottom_; ///< Index for pushing (incremented by
                                        ///< owner), acts as boundary for pop.

  /**
   * @brief Calculates the smallest power of 2 >= requested capacity.
   * @param requested The desired minimum capacity.
   * @return The calculated power-of-2 capacity.
   * @note Marked private as intended.
   */
  static std::size_t calculate_capacity(std::size_t requested) {
    if (requested < 2)
      return 2; // Minimum capacity of 2
    std::size_t p = 1;
    // Find the smallest power of 2 >= requested
    // Check p != 0 to prevent infinite loop in case of overflow, though
    // unlikely with size_t
    while (p < requested && p != 0) {
      p <<= 1;
    }
    // If p overflowed (became 0), return the largest possible power of 2 for
    // size_t This is highly unlikely for typical capacities.
    if (p == 0)
      return (std::size_t(1) << (sizeof(std::size_t) * 8 - 1));
    return p;
  }

public:
  /**
   * @brief Constructs the Chase-Lev deque.
   * @param initial_capacity The desired initial capacity. Will be rounded up to
   * the nearest power of 2 (min 2).
   * @throws std::invalid_argument If rounding capacity results in 0 or less
   * than 2.
   * @throws std::bad_alloc If buffer allocation fails.
   */
  explicit ChaseLevDeque(std::size_t initial_capacity = 1024);

  // Rule of 5/6: Ensure proper handling of resources and atomics.
  ~ChaseLevDeque() = default; // unique_ptr handles buffer cleanup.

  // Delete copy operations: copying atomics and buffer state concurrently is
  // unsafe/complex.
  ChaseLevDeque(const ChaseLevDeque &) = delete;
  ChaseLevDeque &operator=(const ChaseLevDeque &) = delete;

  // Allow move operations (with caution: assumes no concurrent access during
  // move).
  ChaseLevDeque(ChaseLevDeque &&other) noexcept;
  ChaseLevDeque &operator=(ChaseLevDeque &&other) noexcept;

  /**
   * @brief Pushes a task onto the bottom of the deque (LIFO for owner).
   *
   * This operation should only be called by the owner thread.
   *
   * @param task The task to push. The task is moved into the deque.
   * @return `true` if the push was successful, `false` if the deque was full.
   */
  bool push_bottom(Task task);

  /**
   * @brief Pops a task from the bottom of the deque (LIFO for owner).
   *
   * This operation should only be called by the owner thread. It handles
   * potential races with stealers trying to take the last element.
   *
   * @return An `std::optional<Task>` containing the popped task if successful,
   *         `std::nullopt` if the deque was empty or the last element was
   * concurrently stolen.
   */
  std::optional<Task> pop_bottom();

  /**
   * @brief Steals a task from the top of the deque (FIFO for thieves).
   *
   * This operation can be called by any thread (typically thief threads).
   *
   * @return An `std::optional<Task>` containing the stolen task if successful,
   *         `std::nullopt` if the deque was empty or another operation
   * contended for the same element.
   */
  std::optional<Task> steal_top();

  /**
   * @brief Estimates the current number of tasks in the deque.
   * @warning This is an estimate and may be inaccurate under high contention
   * due to the non-atomic nature of reading `top` and `bottom` together.
   * @return The estimated number of tasks.
   */
  std::size_t estimate_size() const;

  /**
   * @brief Checks if the deque is likely empty.
   * @warning Like `estimate_size`, this check can be slightly delayed or
   * inaccurate under high contention.
   * @return `true` if the deque appears empty, `false` otherwise.
   */
  bool is_empty() const;

  /**
   * @brief Gets the fixed capacity of the deque.
   * @return The capacity (power of 2).
   */
  std::size_t capacity() const { return capacity_; }
};

//----------------------------------------------------------------------------
// Global State for Work Stealing Scheduler
//----------------------------------------------------------------------------

extern std::atomic<size_t> g_pending_tasks_ws; ///< Global counter for pending
                                               ///< tasks in work-stealing mode.
extern std::atomic<bool>
    g_all_tasks_submitted_ws; ///< Flag indicating if initial task submission is
                              ///< complete.

//----------------------------------------------------------------------------
// Dynamic Scheduler Worker and Execution Functions
//----------------------------------------------------------------------------

/**
 * @brief Worker function for the simple dynamic scheduler (using TaskQueue).
 * @param thread_id The ID of this worker thread.
 * @param queue The shared TaskQueue instance.
 * @param results_out Shared vector for storing results (atomically updated).
 */
void dynamic_worker(int thread_id, TaskQueue &queue,
                    std::vector<RangeResult> &results_out);

/**
 * @brief Worker function for the work-stealing dynamic scheduler (using
 * ChaseLevDeque).
 *
 * Implements the work-stealing logic: tries to pop from the local deque,
 * otherwise attempts to steal from other deques. Uses global atomics for
 * termination detection.
 *
 * @param thread_id The ID of this worker thread (index into queues vector).
 * @param num_threads Total number of worker threads.
 * @param queues Vector containing all ChaseLevDeque instances (one per thread).
 * Reference must be non-const for stealing.
 * @param results_out Shared vector for storing results (atomically updated).
 */
void dynamic_work_stealing_worker(
    int thread_id, int num_threads,
    std::vector<ChaseLevDeque> &queues, // Uses ChaseLevDeque
    std::vector<RangeResult> &results_out);

/**
 * @brief Executes the Collatz computation using the simple dynamic task queue
 * scheduler.
 * @param config Configuration parameters (num_threads, chunk_size, ranges,
 * etc.).
 * @param results_out Vector to store the final results.
 * @return True if successful, false on configuration error or execution
 * failure.
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out);

/**
 * @brief Executes the Collatz computation using the dynamic work-stealing
 * scheduler with ChaseLevDeques.
 *
 * Creates per-thread ChaseLevDeques, spawns workers that implement the
 * work-stealing strategy (pop local, steal remote), and uses global atomics for
 * termination.
 *
 * @param config Configuration parameters (num_threads, chunk_size, ranges,
 * etc.).
 * @param results_out Vector to store the final results.
 * @return True if successful, false on configuration error or execution
 * failure.
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out);

#endif // DYNAMIC_SCHEDULER_H
