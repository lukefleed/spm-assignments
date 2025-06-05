#ifndef SEQUENTIAL_MERGESORT_HPP
#define SEQUENTIAL_MERGESORT_HPP

#include "../common/record.hpp"
#include <vector>

/**
 * @brief Sorts a vector of Records using a sequential, recursive merge sort.
 * @param data The vector of Records to be sorted in-place.
 */
void sequential_mergesort(std::vector<Record> &data);

#endif // SEQUENTIAL_MERGESORT_HPP
