#include "sequential.h"
#include "collatz.h" // Per find_max_steps_in_subrange

std::vector<ull> run_sequential(const std::vector<Range> &ranges) {
  std::vector<ull> results;
  results.reserve(ranges.size());

  for (const auto &range : ranges) {
    results.push_back(find_max_steps_in_subrange(range.start, range.end));
  }

  return results;
}
