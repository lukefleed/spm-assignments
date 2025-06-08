#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief High-performance parallel merge sort using FastFlow framework
 *
 * Implements optimized three-phase merge sort algorithm for large datasets:
 * 1. Parallel initial sorting of cache-friendly chunks (1024+ elements)
 * 2. Iterative parallel merge passes with buffer ping-ponging strategy
 * 3. Final data placement ensuring in-place result semantics
 *
 * Algorithm characteristics:
 * - Time complexity: O(n log n) with parallel speedup potential
 * - Space complexity: O(n) auxiliary buffer for out-of-place merging
 * - Thread safety: Full parallelization through FastFlow farm patterns
 * - Exception safety: Strong guarantee with automatic resource cleanup
 *
 * Performance optimizations:
 * - Sequential fallback for datasets < (num_threads * 1024) elements
 * - 4x oversubscription for load balancing resilience
 * - Move semantics to minimize Record copying overhead
 * - Cache-optimized chunk sizes for initial sorting phase
 *
 * @param data Input vector sorted in-place (modified on completion)
 * @param num_threads Worker thread count (0 defaults to single-threaded)
 *
 * @throws std::runtime_error if FastFlow farm execution fails
 * @throws std::bad_alloc if auxiliary buffer allocation fails
 *
 * @pre data.size() fits in size_t range
 * @post data is sorted in ascending order by Record::key
 * @post Strong exception safety: data unchanged if exception thrown
 */
void parallel_mergesort(std::vector<Record> &data, const size_t num_threads);

#endif // FF_MERGESORT_HPP
