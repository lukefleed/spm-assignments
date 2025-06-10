#ifndef SEQUENTIAL_MERGESORT_HPP
#define SEQUENTIAL_MERGESORT_HPP

#include "../common/record.hpp"
#include <vector>

/**
 * @brief Sequential recursive merge sort implementation
 * @param data Vector of records to sort in-place by key
 */
void sequential_mergesort(std::vector<Record> &data);

#endif // SEQUENTIAL_MERGESORT_HPP
