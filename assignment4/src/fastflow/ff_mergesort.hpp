#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief Parallel merge sort using FastFlow framework
 * @param data Vector to sort in-place by key
 * @param num_threads Worker thread count
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads);

#endif // FF_MERGESORT_HPP
