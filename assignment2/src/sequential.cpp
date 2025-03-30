/**
 * @file sequential.cpp
 * @brief Implementation of the sequential version of the Collatz conjecture
 * problem
 */
#include "sequential.h"
#include "collatz.h" // For find_max_steps_in_subrange

/**
 * @brief Runs the Collatz conjecture calculation sequentially on multiple
 * ranges
 *
 * @param ranges Vector of Range objects defining start and end points for
 * calculations
 * @return std::vector<ull> Vector containing the maximum number of steps for
 * each range
 */
std::vector<ull> run_sequential(const std::vector<Range> &ranges) {
  std::vector<ull> results;
  results.reserve(ranges.size());

  for (const auto &range : ranges) {
    results.push_back(find_max_steps_in_subrange(range.start, range.end));
  }

  return results;
}
