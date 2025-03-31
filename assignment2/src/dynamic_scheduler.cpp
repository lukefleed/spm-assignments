#include "dynamic_scheduler.h"
#include "collatz.h" // For find_max_steps_in_subrange
#include <algorithm> // For std::min, std::max
#include <atomic>
#include <deque>    // Added for TaskQueue implementation
#include <iostream> // For verbose/error output
#include <limits>   // For std::numeric_limits
#include <optional>
#include <stdexcept> // For std::invalid_argument in ChaseLevDeque constructor
#include <thread>
#include <vector>

//----------------------------------------------------------------------------
// TaskQueue (Simple Centralized Queue - Implementation)
//----------------------------------------------------------------------------

// Implementations for TaskQueue::push, pop, close (identical to previous
// version) Kept here for completeness if the TaskQueue option is retained.

void TaskQueue::push(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_.load(std::memory_order_relaxed)) { // Use load for atomic bool
      return;
    }
    queue_.push_back(std::move(task)); // Use push_back for deque
  }
  cond_var_.notify_one();
}

std::optional<Task> TaskQueue::pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.wait(lock, [this] {
    return !queue_.empty() || closed_.load(std::memory_order_relaxed);
  });

  if (queue_.empty()) {
    // Check closed_ again after wait confirms the predicate.
    if (closed_.load(std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return std::nullopt; // Should not happen if predicate is correct
  }

  Task task = std::move(queue_.front());
  queue_.pop_front(); // Use pop_front for deque
  return task;
}

void TaskQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_.store(true, std::memory_order_relaxed); // Use store for atomic bool
  }
  cond_var_.notify_all();
}

//----------------------------------------------------------------------------
// ChaseLevDeque (Lock-Free Work-Stealing Deque - Implementation)
//----------------------------------------------------------------------------

/**
 * @brief Constructs the Chase-Lev deque.
 * @param initial_capacity The desired initial capacity. Will be rounded up to
 * the nearest power of 2 (min 2).
 * @throws std::invalid_argument If rounding capacity results in 0 or less
 * than 2.
 * @throws std::bad_alloc If buffer allocation fails.
 */
ChaseLevDeque::ChaseLevDeque(std::size_t initial_capacity)
    : // --- Initializer List Order Corrected ---
      capacity_(
          calculate_capacity(initial_capacity)), // Calculate capacity first
      buffer_(nullptr), // Initialize buffer_ after capacity_
      top_(0), bottom_(0) {
  // Validate capacity *after* calculation
  if (capacity_ < 2 ||
      capacity_ > (std::size_t(1) << (sizeof(std::size_t) * 8 - 2))) {
    throw std::invalid_argument("ChaseLevDeque capacity must be a power of 2, "
                                ">= 2, and within reasonable limits.");
  }
  // Allocate the buffer using make_unique
  buffer_ = std::make_unique<Task[]>(capacity_);
}

// Move constructor
ChaseLevDeque::ChaseLevDeque(ChaseLevDeque &&other) noexcept
    : capacity_(other.capacity_),        // capacity_ is const, copy its value
      buffer_(std::move(other.buffer_)), // Move the buffer ownership
      top_(other.top_.load(std::memory_order_relaxed)),
      bottom_(other.bottom_.load(std::memory_order_relaxed)) {}

// Move assignment operator
ChaseLevDeque &ChaseLevDeque::operator=(ChaseLevDeque &&other) noexcept {
  if (this != &other) {
    // Check compatibility. Since capacity_ is const, it cannot be reassigned.
    // We rely on the check being implicitly handled by the fact that move
    // assignment should typically only happen between objects initially created
    // compatibly, or we accept that assigning between different capacities is
    // problematic. A runtime check `if (capacity_ != other.capacity_)` could be
    // added, but throwing violates noexcept.
    buffer_ = std::move(other.buffer_);
    top_.store(other.top_.load(std::memory_order_relaxed),
               std::memory_order_relaxed);
    bottom_.store(other.bottom_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  }
  return *this;
}

/**
 * @brief Pushes a task onto the bottom of the deque (LIFO for owner).
 * (Implementation identical to previous corrected version)
 */
bool ChaseLevDeque::push_bottom(Task task) {
  std::size_t b = bottom_.load(std::memory_order_relaxed);
  std::size_t t = top_.load(std::memory_order_acquire);
  if (b - t >= capacity_) {
    return false; // Full
  }
  buffer_[b & (capacity_ - 1)] = std::move(task);
  bottom_.store(b + 1, std::memory_order_release);
  return true;
}

/**
 * @brief Pops a task from the bottom of the deque (LIFO for owner).
 * (Implementation identical to previous corrected version)
 */
std::optional<Task> ChaseLevDeque::pop_bottom() {
  std::size_t b = bottom_.load(std::memory_order_relaxed);
  if (b == 0)
    return std::nullopt;
  b--;
  bottom_.store(b, std::memory_order_seq_cst);
  std::size_t t = top_.load(std::memory_order_seq_cst);
  long size = static_cast<long>(b) - static_cast<long>(t);

  if (size < 0) {
    bottom_.store(t, std::memory_order_relaxed);
    return std::nullopt;
  }
  Task task = std::move(buffer_[b & (capacity_ - 1)]);
  if (size > 0) {
    return task;
  }
  if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                    std::memory_order_relaxed)) {
    bottom_.store(t + 1, std::memory_order_relaxed);
    return std::nullopt;
  } else {
    bottom_.store(t + 1, std::memory_order_relaxed);
    return task;
  }
}

/**
 * @brief Steals a task from the top of the deque (FIFO for thieves).
 * (Implementation identical to previous corrected version)
 */
std::optional<Task> ChaseLevDeque::steal_top() {
  std::size_t t = top_.load(std::memory_order_seq_cst);
  std::size_t b = bottom_.load(std::memory_order_seq_cst);
  if (static_cast<long>(t) >= static_cast<long>(b)) {
    return std::nullopt;
  }
  // Read buffer content *after* checking bounds, but potentially before CAS
  // If Task had non-trivial copy/move, might read after CAS success
  Task task =
      std::move(buffer_[t & (capacity_ - 1)]); // Use move if Task allows

  if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                    std::memory_order_relaxed)) {
    return std::nullopt;
  }
  // CAS succeeded, the read/move we did (or do now) is valid
  return task;
}

/**
 * @brief Estimates the current number of tasks in the deque.
 * (Implementation identical to previous corrected version)
 */
std::size_t ChaseLevDeque::estimate_size() const {
  std::size_t b = bottom_.load(std::memory_order_relaxed);
  std::size_t t = top_.load(std::memory_order_relaxed);
  return (b >= t) ? (b - t) : 0;
}

/**
 * @brief Checks if the deque is likely empty.
 * (Implementation identical to previous corrected version)
 */
bool ChaseLevDeque::is_empty() const {
  std::size_t b = bottom_.load(std::memory_order_relaxed);
  std::size_t t = top_.load(std::memory_order_relaxed);
  return (b <= t);
}

//----------------------------------------------------------------------------
// Global State & Worker Functions (Work Stealing - Using ChaseLevDeque)
//----------------------------------------------------------------------------

// Define global atomics (as declared extern in header)
std::atomic<size_t> g_pending_tasks_ws{0};
std::atomic<bool> g_all_tasks_submitted_ws{false};

/**
 * @brief Worker function for the work-stealing dynamic scheduler (using
 * ChaseLevDeque). (Implementation identical to previous corrected version,
 * including backoff)
 */
void dynamic_work_stealing_worker(
    int thread_id, int num_threads,
    std::vector<ChaseLevDeque> &queues, // Uses ChaseLevDeque
    std::vector<RangeResult> &results_out) {

  thread_local std::vector<int> failed_steal_attempts(num_threads, 0);
  thread_local int consecutive_idle_loops = 0;
  const int MAX_IDLE_LOOPS_BEFORE_SLEEP = 100;
  const int BASE_BACKOFF_MS = 1;
  const int MAX_BACKOFF_MS = 64;

  while (true) {
    std::optional<Task> task_opt;
    task_opt = queues[thread_id].pop_bottom();

    if (!task_opt) {
      int victim_start_offset = 1;
      int attempts = 0;
      while (attempts < num_threads - 1) {
        int victim_id =
            (thread_id + victim_start_offset + attempts) % num_threads;
        if (victim_id == thread_id) {
          attempts++;
          continue;
        }
        task_opt = queues[victim_id].steal_top();
        if (task_opt) {
          consecutive_idle_loops = 0;
          failed_steal_attempts[victim_id] = 0;
          break;
        } else {
          failed_steal_attempts[victim_id]++;
        }
        attempts++;
      }
    }

    if (task_opt) {
      consecutive_idle_loops = 0;
      Task task = *task_opt;
      ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

      if (local_max_steps > 0) {
        ull current_max = results_out[task.original_range_index].max_steps.load(
            std::memory_order_relaxed);
        while (local_max_steps > current_max) {
          if (results_out[task.original_range_index]
                  .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed)) {
            break;
          }
        }
      }
      g_pending_tasks_ws.fetch_sub(1, std::memory_order_release);

    } else {
      if (g_all_tasks_submitted_ws.load(std::memory_order_acquire) &&
          g_pending_tasks_ws.load(std::memory_order_acquire) == 0) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (g_pending_tasks_ws.load(std::memory_order_relaxed) == 0) {
          break;
        }
      }

      consecutive_idle_loops++;
      if (consecutive_idle_loops < MAX_IDLE_LOOPS_BEFORE_SLEEP) {
        std::this_thread::yield();
      } else {
        int backoff_ms = std::min(
            MAX_BACKOFF_MS, BASE_BACKOFF_MS << (consecutive_idle_loops /
                                                MAX_IDLE_LOOPS_BEFORE_SLEEP));
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      }
    }
  } // End while(true) worker loop
}

//----------------------------------------------------------------------------
// Dynamic Scheduler Execution Functions (Work Stealing - Using ChaseLevDeque)
//----------------------------------------------------------------------------

/**
 * @brief Executes the Collatz computation using the dynamic work-stealing
 * scheduler with ChaseLevDeques. (Modified verbose output and `attempts` type)
 */
bool run_dynamic_work_stealing(const Config &config,
                               std::vector<RangeResult> &results_out) {
  // --- Validate Configuration ---
  if (config.num_threads <= 0) {
    std::cerr
        << "Error: Dynamic work stealing requires a positive number of threads."
        << std::endl;
    return false;
  }
  if (config.chunk_size == 0) {
    std::cerr << "Error: Dynamic work stealing requires a positive chunk size "
                 "(> 0) for task generation."
              << std::endl;
    return false;
  }

  // --- Reset Global State ---
  g_pending_tasks_ws.store(0, std::memory_order_relaxed);
  g_all_tasks_submitted_ws.store(false, std::memory_order_relaxed);

  // --- Create Deques ---
  std::vector<ChaseLevDeque> queues;
  size_t estimated_total_tasks = 0;
  for (const auto &range : config.ranges) {
    if (range.start <= range.end) {
      ull range_len = range.end - range.start + 1;
      // Avoid overflow potential if range_len is huge and chunk_size is 1
      if (config.chunk_size == 0)
        continue; // Already checked, but defensive
      ull num_tasks_in_range =
          (range_len > 0)
              ? (range_len + config.chunk_size - 1) / config.chunk_size
              : 0;
      // Check for overflow before adding
      if (std::numeric_limits<size_t>::max() - estimated_total_tasks <
          num_tasks_in_range) {
        estimated_total_tasks = std::numeric_limits<size_t>::max();
        break;
      } else {
        estimated_total_tasks += num_tasks_in_range;
      }
    }
  }
  size_t base_capacity = 256;
  size_t capacity_per_deque_calc =
      base_capacity + ((estimated_total_tasks / config.num_threads) * 2);
  capacity_per_deque_calc =
      std::max<size_t>(16, std::min<size_t>(65536, capacity_per_deque_calc));
  // Use the static method locally to get the final power-of-2 size for the
  // message size_t final_rounded_capacity =
  // ChaseLevDeque::calculate_capacity(capacity_per_deque_calc); // Error:
  // private method

  // We need to instantiate one deque to get its actual capacity for the message
  size_t actual_deque_capacity = 0;
  try {
    // Use emplace_back which calls the constructor directly.
    for (unsigned int i = 0; i < config.num_threads; ++i) {
      queues.emplace_back(capacity_per_deque_calc);
      if (i == 0)
        actual_deque_capacity =
            queues[0].capacity(); // Get actual capacity from first instance
    }
  } catch (const std::exception &e) {
    std::cerr << "Error creating ChaseLevDeques: " << e.what() << std::endl;
    return false;
  }

  if (config.verbose) {
    std::cout << "Initializing " << config.num_threads
              << " ChaseLevDeques with actual capacity: "
              << actual_deque_capacity << std::endl;
  }

  std::vector<std::thread> threads;

  // --- Initialize Results Vector ---
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r);
  }

  // --- Launch Worker Threads ---
  threads.reserve(config.num_threads);
  for (unsigned int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(dynamic_work_stealing_worker, i, config.num_threads,
                         std::ref(queues), std::ref(results_out));
  }

  // --- Main Thread: Initial Task Distribution ---
  size_t current_queue_idx = 0;
  size_t total_tasks_pushed = 0;
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue;

    ull current_start = range.start;
    while (current_start <= range.end) {
      ull current_chunk_end = (current_start > std::numeric_limits<ull>::max() -
                                                   (config.chunk_size - 1))
                                  ? range.end
                                  : current_start + config.chunk_size - 1;
      ull current_end = std::min(range.end, current_chunk_end);

      g_pending_tasks_ws.fetch_add(1, std::memory_order_relaxed);
      Task current_task = {current_start, current_end, i};

      // --- Corrected `attempts` type ---
      size_t attempts = 0; // Use size_t for comparison with unsigned int
      while (!queues[current_queue_idx].push_bottom(current_task)) {
        attempts++;
        // Compare size_t with unsigned int - OK
        if (attempts >= config.num_threads) {
          if (config.verbose && (attempts % config.num_threads == 0)) {
            std::cerr << "Warning: All deques appear full during initial "
                         "distribution. Yielding and retrying queue "
                      << current_queue_idx << "..." << std::endl;
          }
          std::this_thread::yield();
          attempts = 0; // Reset attempts, retry same queue index after yield
        } else {
          current_queue_idx = (current_queue_idx + 1) % config.num_threads;
        }
      }
      total_tasks_pushed++;
      current_queue_idx = (current_queue_idx + 1) % config.num_threads;

      if (current_end == range.end)
        break;
      if (current_end == std::numeric_limits<ull>::max())
        break;
      current_start = current_end + 1;
    }
  }

  if (config.verbose) {
    std::cout << "Initial task distribution complete. Pushed "
              << total_tasks_pushed << " tasks." << std::endl;
  }

  g_all_tasks_submitted_ws.store(true, std::memory_order_release);

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  size_t final_pending = g_pending_tasks_ws.load(std::memory_order_relaxed);
  if (config.verbose && final_pending != 0) {
    std::cerr << "Warning: Pending tasks counter is non-zero (" << final_pending
              << ") after joining threads. Expected 0." << std::endl;
  }

  return true;
}

// --- Implementation for run_dynamic_task_queue and dynamic_worker ---
// (Identical to previous corrected version, kept if TaskQueue option is
// desired)

bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out) {
  if (config.num_threads <= 0 || config.chunk_size == 0) {
    std::cerr << "Error: Dynamic task queue requires positive num_threads and "
                 "chunk_size."
              << std::endl;
    return false;
  }
  TaskQueue task_queue;
  std::vector<std::thread> threads;
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r);
  }
  threads.reserve(config.num_threads);
  for (unsigned int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(dynamic_worker, i, std::ref(task_queue),
                         std::ref(results_out));
  }
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    if (range.start > range.end)
      continue;
    ull current_start = range.start;
    while (current_start <= range.end) {
      ull current_chunk_end = (current_start > std::numeric_limits<ull>::max() -
                                                   (config.chunk_size - 1))
                                  ? range.end
                                  : current_start + config.chunk_size - 1;
      ull current_end = std::min(range.end, current_chunk_end);
      task_queue.push({current_start, current_end, i});
      if (current_end == range.end)
        break;
      if (current_end == std::numeric_limits<ull>::max())
        break;
      current_start = current_end + 1;
    }
  }
  task_queue.close();
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  return true;
}

void dynamic_worker(int thread_id [[maybe_unused]], TaskQueue &queue,
                    std::vector<RangeResult> &results_out) {
  while (true) {
    std::optional<Task> task_opt = queue.pop();
    if (!task_opt)
      break;
    Task task = *task_opt;
    ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);
    if (local_max_steps > 0) {
      ull current_max = results_out[task.original_range_index].max_steps.load(
          std::memory_order_relaxed);
      while (local_max_steps > current_max) {
        if (results_out[task.original_range_index]
                .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
          break;
        }
      }
    }
  }
}
