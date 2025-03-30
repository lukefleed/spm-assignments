#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <atomic>
#include <optional>
#include <string>
#include <vector>

// Alias for unsigned long long for clarity.
using ull = unsigned long long;

// Structure to represent a range.
struct Range {
  ull start;
  ull end;
};

// Structure to represent a task (used in the dynamic scheduler).
struct Task {
  ull start;
  ull end;
  size_t original_range_index; // Index of the original range from which the
                               // task originates.
};

// Structure to store the result (maximum found) for a range, using std::atomic
// for thread-safe updates.
struct RangeResult {
  Range original_range;
  std::atomic<ull> max_steps{0}; // Initialized to 0.

  // Constructor to initialize the original range.
  RangeResult(const Range &r) : original_range(r) {}

  // Copy constructor required for using RangeResult in
  // std::vector<RangeResult>.
  RangeResult(const RangeResult &other)
      : original_range(other.original_range),
        max_steps(other.max_steps.load()) {}

  // Assignment operator required for using RangeResult in
  // std::vector<RangeResult>.
  RangeResult &operator=(const RangeResult &other) {
    if (this != &other) {
      original_range = other.original_range;
      max_steps.store(other.max_steps.load());
    }
    return *this;
  }

  // Default constructor required for std::vector::emplace_back.
  RangeResult() = default;
};

// Structure to hold the program configuration read from the arguments.
enum class SchedulingType { SEQUENTIAL, STATIC, DYNAMIC };

// Enum for static scheduling variants.
enum class StaticVariant { BLOCK, CYCLIC, BLOCK_CYCLIC };

struct Config {
  SchedulingType scheduling = SchedulingType::STATIC; // Default is static.
  StaticVariant static_variant =
      StaticVariant::BLOCK_CYCLIC; // Default is block-cyclic.
  int num_threads = 16;            // Default number of threads.
  ull chunk_size = 1;              // Default chunk/block size.
  std::vector<Range> ranges;
  bool verbose = false; // Optional for debug.
};

#endif // COMMON_TYPES_H
