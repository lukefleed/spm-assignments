#include "sequential_mergesort.hpp"
#include <algorithm>
#include <vector>

/**
 * @brief Merges two sorted sub-arrays into a single sorted array.
 * @param data The main vector containing the sub-arrays.
 * @param temp A temporary vector used for storing the merged result.
 * @param left The starting index of the first sub-array.
 * @param mid The ending index of the first sub-array.
 * @param right The ending index of the second sub-array.
 */
static void merge(std::vector<Record> &data, std::vector<Record> &temp,
                  size_t left, size_t mid, size_t right) {
  size_t i = left, j = mid + 1, k = left;

  // Merge the two parts into the temporary vector
  while (i <= mid && j <= right) {
    if (data[i] <= data[j]) {
      temp[k++] = std::move(data[i++]);
    } else {
      temp[k++] = std::move(data[j++]);
    }
  }

  // Copy remaining elements of the left half
  while (i <= mid) {
    temp[k++] = std::move(data[i++]);
  }

  // Copy remaining elements of the right half
  while (j <= right) {
    temp[k++] = std::move(data[j++]);
  }

  // Move the sorted elements from temp back to the original vector
  for (i = left; i <= right; ++i) {
    data[i] = std::move(temp[i]);
  }
}

/**
 * @brief The recursive core of the merge sort algorithm.
 * @param data The vector to sort.
 * @param temp A temporary buffer for merging.
 * @param left The starting index of the segment to sort.
 * @param right The ending index of the segment to sort.
 */
static void mergesort_recursive(std::vector<Record> &data,
                                std::vector<Record> &temp, size_t left,
                                size_t right) {
  if (left < right) {
    size_t mid = left + (right - left) / 2;
    mergesort_recursive(data, temp, left, mid);
    mergesort_recursive(data, temp, mid + 1, right);
    merge(data, temp, left, mid, right);
  }
}

void sequential_mergesort(std::vector<Record> &data) {
  if (data.size() <= 1) {
    return;
  }
  std::vector<Record> temp(data.size());
  mergesort_recursive(data, temp, 0, data.size() - 1);
}
