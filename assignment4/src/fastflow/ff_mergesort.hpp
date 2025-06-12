#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief Sorts a vector of Records in parallel using a pipelined FastFlow
 * mergesort implementation.
 *
 * This implementation uses an ff_pipeline to orchestrate the sorting process.
 * An initial stage uses an ff_farm to sort sub-chunks of the data concurrently.
 * A subsequent stage iteratively merges these sorted chunks in parallel, also
 * using an ff_farm within a loop, until the entire vector is sorted. This
 * approach encapsulates the entire logic within a single, integrated parallel
 * execution flow.
 *
 * @param data The vector of Records to be sorted. The vector will be modified
 * in place.
 * @param num_threads The number of threads to use for the parallel operations.
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads);

#endif
