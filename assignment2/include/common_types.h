#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <atomic>   // For std::atomic<ull> used in RangeResult
#include <optional> // Potentially useful for functions returning optional values (not directly used here)
#include <string>   // Potentially useful (not directly used here)
#include <vector>   // For std::vector<Range> in Config

/** @brief Type alias for unsigned long long, commonly used for large integer
 * ranges and counts. */
using ull = unsigned long long;

/**
 * @brief Represents a continuous range of unsigned long long integers.
 */
struct Range {
  ull start; /**< The starting value of the range (inclusive). */
  ull end;   /**< The ending value of the range (inclusive). */
};

/**
 * @brief Represents a unit of work (a sub-range) for the dynamic scheduler.
 */
struct Task {
  ull start; /**< The starting value of the task's sub-range (inclusive). */
  ull end;   /**< The ending value of the task's sub-range (inclusive). */
  /**
   * @brief Index referencing the original `Range` in the `Config::ranges`
   * vector from which this task was derived. Needed to update the correct
   * result entry.
   */
  size_t original_range_index;
};

/**
 * @brief Stores the result (maximum steps) for an original input `Range`.
 *        Uses `std::atomic` to allow thread-safe updates from multiple worker
 * threads.
 */
struct RangeResult {
  Range original_range; /**< A copy of the original input range this result
                           corresponds to. */
  /**
   * @brief The maximum Collatz steps found within the `original_range`.
   *        `std::atomic` ensures that concurrent updates (e.g., using
   * `fetch_max` or compare-and-swap loops) are handled correctly without data
   * races. Initialized to 0. Relaxed memory order is often sufficient for
   * updates.
   */
  std::atomic<ull> max_steps{0};

  /** @brief Constructor to initialize from an original Range. */
  explicit RangeResult(const Range &r)
      : original_range(r), max_steps(0) {
  } // Explicit to prevent accidental conversions.

  /**
   * @brief Copy constructor. Necessary for storing `RangeResult` in standard
   * containers like `std::vector`. Performs an atomic load to copy the
   * `max_steps` value.
   */
  RangeResult(const RangeResult &other)
      : original_range(other.original_range),
        max_steps(other.max_steps.load(std::memory_order_relaxed)) {
  } // Relaxed load is sufficient for copying the value.

  /**
   * @brief Copy assignment operator. Also required for container compatibility.
   *        Performs an atomic load and store.
   */
  RangeResult &operator=(const RangeResult &other) {
    if (this != &other) { // Protect against self-assignment.
      original_range = other.original_range;
      max_steps.store(other.max_steps.load(std::memory_order_relaxed),
                      std::memory_order_relaxed); // Relaxed load/store.
    }
    return *this;
  }

  /**
   * @brief Move constructor. Handles moving resources if RangeResult had
   * complex members. For atomic, we still load the value from the source.
   */
  RangeResult(RangeResult &&other) noexcept
      : original_range(std::move(other.original_range)),
        max_steps(other.max_steps.load(std::memory_order_relaxed)) {
    // Optionally, reset the source atomic if required, though often not
    // necessary after move. other.max_steps.store(0,
    // std::memory_order_relaxed);
  }

  /**
   * @brief Move assignment operator. Handles moving resources.
   */
  RangeResult &operator=(RangeResult &&other) noexcept {
    if (this != &other) {
      original_range = std::move(other.original_range);
      max_steps.store(other.max_steps.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      // Optionally reset source.
      // other.max_steps.store(0, std::memory_order_relaxed);
    }
    return *this;
  }

  /** @brief Default constructor. May be needed by containers in some
   * situations. */
  RangeResult() = default;
};

/** @brief Enumerates the high-level scheduling approaches. */
enum class SchedulingType {
  SEQUENTIAL, /**< Single-threaded execution. */
  STATIC,     /**< Work distribution decided before execution (Block, Cyclic,
                 Block-Cyclic). */
  DYNAMIC     /**< Work distribution adapted during execution (Task Queue, Work
                 Stealing). */
};

/** @brief Enumerates the specific variants for static scheduling. */
enum class StaticVariant {
  BLOCK,       /**< Divide work into N contiguous blocks, one per thread. */
  CYCLIC,      /**< Assign work units (e.g., single numbers) round-robin. */
  BLOCK_CYCLIC /**< Divide work into small blocks, assign blocks round-robin. */
};

/**
 * @brief Holds the application's configuration, typically parsed from
 * command-line arguments.
 */
struct Config {
  /** @brief The primary scheduling method to use. */
  SchedulingType scheduling =
      SchedulingType::SEQUENTIAL; // Default to sequential unless threads > 1
                                  // specified.

  /** @brief The specific variant if `scheduling` is STATIC. */
  StaticVariant static_variant =
      StaticVariant::BLOCK_CYCLIC; // A common default balancing load/locality.

  /** @brief Number of worker threads to use for parallel execution. Default 1
   * means sequential. */
  unsigned int num_threads =
      1; // Defaulting to 1 ensures sequential execution if not overridden.

  /**
   * @brief Size of work units for certain schedulers.
   *        - For STATIC BLOCK_CYCLIC: Size of the blocks assigned cyclically.
   *        - For DYNAMIC: Size of tasks generated from ranges.
   *        Defaulting to 64 is often a reasonable starting point related to
   * cache line sizes, but optimal value is application/hardware dependent.
   */
  ull chunk_size = 64;

  /** @brief Vector storing the input ranges provided by the user. */
  std::vector<Range> ranges;

  /** @brief Flag to enable verbose diagnostic output during execution. */
  bool verbose = false;
};

#endif // COMMON_TYPES_H
