#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief Parallel merge sort implementation using FastFlow framework
 *
 * Implements a multi-stage parallel merge sort algorithm with synchronized
 * farm patterns. The algorithm divides the input into fixed-size chunks for
 * initial sorting, followed by iterative parallel merge phases until the
 * entire dataset is sorted.
 *
 * @param data Input vector of Record objects to sort in-place
 * @param num_threads Number of worker threads for parallel execution
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads);

#endif // FF_MERGESORT_HPP
