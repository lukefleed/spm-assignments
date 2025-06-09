#ifndef FF_MERGESORT_HPP
#define FF_MERGESORT_HPP

#include "../common/record.hpp"
#include <cstddef>
#include <vector>

/**
 * @brief parallel merge sort using FastFlow framework
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
