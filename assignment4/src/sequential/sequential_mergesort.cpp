#include "sequential_mergesort.hpp"
#include <algorithm>
#include <vector>

/**
 * @brief Merge two sorted sub-arrays into single sorted array
 * @param data Main vector containing sub-arrays
 * @param temp Temporary buffer for merge operation
 * @param left Start index of first sub-array
 * @param mid End index of first sub-array
 * @param right End index of second sub-array
 */
static void merge(std::vector<Record> &data, std::vector<Record> &temp,
                  size_t left, size_t mid, size_t right) {
  size_t i = left, j = mid + 1, k = left;

  // Merge both sub-arrays in sorted order
  while (i <= mid && j <= right) {
    if (data[i] <= data[j]) {
      temp[k++] = std::move(data[i++]);
    } else {
      temp[k++] = std::move(data[j++]);
    }
  }

  // Copy remaining elements from left sub-array
  while (i <= mid) {
    temp[k++] = std::move(data[i++]);
  }

  // Copy remaining elements from right sub-array
  while (j <= right) {
    temp[k++] = std::move(data[j++]);
  }

  // Copy merged result back to original array
  for (i = left; i <= right; ++i) {
    data[i] = std::move(temp[i]);
  }
}

/**
 * @brief Recursive merge sort implementation
 * @param data Vector to sort
 * @param temp Temporary buffer for merging
 * @param left Start index of segment
 * @param right End index of segment
 */
static void mergesort_recursive(std::vector<Record> &data,
                                std::vector<Record> &temp, size_t left,
                                size_t right) {
  if (left < right) {
    size_t mid = left + (right - left) / 2; // Avoid overflow
    mergesort_recursive(data, temp, left, mid);
    mergesort_recursive(data, temp, mid + 1, right);
    merge(data, temp, left, mid, right);
  }
}

/**
 * @brief Sequential merge sort entry point
 */
void sequential_mergesort(std::vector<Record> &data) {
  if (data.size() <= 1) {
    return; // Already sorted or empty
  }
  std::vector<Record> temp(data.size());
  mergesort_recursive(data, temp, 0, data.size() - 1);
}
