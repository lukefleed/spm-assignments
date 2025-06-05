#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief Sorts a vector of Records in parallel using a highly-optimized,
 *        pipelined merge sort architecture built with FastFlow.
 *
 * @note The implementation uses a scalable parallel reduction pattern with a
 *       dynamic pipeline of merge farms, which is more advanced than a simple
 *       two-farm setup. The name is kept for API compatibility.
 *
 * @param data The vector of Records to be sorted. The vector is sorted
 * in-place.
 * @param num_threads The total number of threads to utilize for the sorting
 * process. If 0, it defaults to a single thread.
 */
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     const size_t num_threads);

#endif // FF_MERGESORT_HPP
