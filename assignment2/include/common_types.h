#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <atomic> // For std::atomic<ull> used in RangeResult
#include <optional>
#include <string>
#include <vector> // For std::vector<Range> in Config

/** Shorthand for unsigned long long - makes working with large numbers cleaner
 */
using ull = unsigned long long;

/**
 * A simple range with start and end values (both inclusive).
 * Used to specify which numbers we want to process.
 */
struct Range {
  ull start; // First number in the range
  ull end;   // Last number in the range
};

/**
 * A unit of work that contains a sub-range for the dynamic scheduler.
 * Each task knows which original range it came from so we can update
 * the correct result when it's done.
 */
struct Task {
  ull start; // First number in this task's sub-range
  ull end;   // Last number in this task's sub-range

  // Index pointing to the original Range in Config::ranges
  // We need this to know which result to update
  size_t original_range_index;
};

/**
 * Stores the result (maximum steps) for an original input Range. Uses
 * atomic to allow thread-safe updates from multiple worker threads.
 */
struct RangeResult {
  Range original_range; // A copy of the original input range this result
                        // corresponds to

  // The maximum Collatz steps found within the original_range
  // atomic ensures that concurrent updates are handled correctly without data
  // races Initialized to 0. Relaxed memory order is sufficient for updates
  std::atomic<ull> max_steps{0};

  // Constructor to initialize from an original Range
  explicit RangeResult(const Range &r)
      : original_range(r), max_steps(0) {
  } // Explicit to prevent accidental conversions

  // Copy constructor - necessary for storing RangeResult in standard containers
  // like vector Performs an atomic load to copy the max_steps value
  RangeResult(const RangeResult &other)
      : original_range(other.original_range),
        max_steps(other.max_steps.load(std::memory_order_relaxed)) {}

  // Copy assignment operator - also required for container compatibility
  // Performs an atomic load and store
  RangeResult &operator=(const RangeResult &other) {
    if (this != &other) { // Protect against self-assignment
      original_range = other.original_range;
      max_steps.store(other.max_steps.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
    }
    return *this;
  }

  // Move constructor - handles moving resources if RangeResult had complex
  // members For atomic, we still load the value from the source
  RangeResult(RangeResult &&other) noexcept
      : original_range(std::move(other.original_range)),
        max_steps(other.max_steps.load(std::memory_order_relaxed)) {}

  // Move assignment operator - handles moving resources
  RangeResult &operator=(RangeResult &&other) noexcept {
    if (this != &other) {
      original_range = std::move(other.original_range);
      max_steps.store(other.max_steps.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
    }
    return *this;
  }

  // Default constructor - may be needed by containers in some situations
  RangeResult() = default;
  // Different ways to schedule work across threads
  enum class SchedulingType {
    SEQUENTIAL, // Single-threaded execution
    STATIC,     // Work distribution decided before execution
    DYNAMIC     // Work distribution adapted during execution
  };

  // Specific variants for static scheduling
  enum class StaticVariant {
    BLOCK,       // Divide work into N contiguous blocks, one per thread
    CYCLIC,      // Assign work units (e.g., single numbers) round-robin
    BLOCK_CYCLIC // Divide work into small blocks, assign blocks round-robin
  };

  /**
   * Holds the application's configuration
   */
  struct Config {
    // The primary scheduling method to use.
    SchedulingType scheduling = SchedulingType::SEQUENTIAL;

    // The specific variant if `scheduling` is STATIC.
    StaticVariant static_variant = StaticVariant::BLOCK_CYCLIC;

    // Number of worker threads to use for parallel execution.
    unsigned int num_threads = 1;

    /**
     * Size of work units for certain schedulers. For STATIC BLOCK_CYCLIC:
     * Size of the blocks assigned cyclically.  For DYNAMIC: Size of tasks
     * generated from ranges. Defaulting to 64 is often a reasonable starting
     * point related to cache line sizes.
     */
    ull chunk_size = 64;

    // Vector storing the input ranges provided by the user.
    std::vector<Range> ranges;

    // Flag to enable verbose diagnostic output during execution.
    bool verbose = false;
  };

#endif // COMMON_TYPES_H
