#ifndef SEQUENTIAL_MERGESORT_HPP
#define SEQUENTIAL_MERGESORT_HPP

#include "../common/record.hpp"
#include <memory>
#include <vector>

/**
 * @brief Sequential merge sort implementation
 */
void sequential_mergesort(std::vector<std::unique_ptr<Record>> &data,
                          size_t left, size_t right);

/**
 * @brief Wrapper for sequential merge sort
 */
void sequential_mergesort(std::vector<std::unique_ptr<Record>> &data);

/**
 * @brief STL sort wrapper for baseline comparison
 */
void stl_sort(std::vector<std::unique_ptr<Record>> &data);

#endif // SEQUENTIAL_MERGESORT_HPP
