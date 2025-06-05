#include "sequential_mergesort.hpp"
#include <algorithm>
#include <vector>

/**
 * @brief Merge two sorted subarrays
 */
static void merge(std::vector<std::unique_ptr<Record>> &data, size_t left,
                  size_t mid, size_t right) {
  size_t n1 = mid - left + 1;
  size_t n2 = right - mid;

  // Temporary storage for pointers only
  std::vector<std::unique_ptr<Record>> temp;
  temp.reserve(n1 + n2);

  // Move elements to temp
  for (size_t i = left; i <= right; ++i) {
    temp.push_back(std::move(data[i]));
  }

  // Merge back
  size_t i = 0, j = n1, k = left;

  while (i < n1 && j < n1 + n2) {
    if (temp[i]->key <= temp[j]->key) {
      data[k++] = std::move(temp[i++]);
    } else {
      data[k++] = std::move(temp[j++]);
    }
  }

  // Copy remaining elements
  while (i < n1) {
    data[k++] = std::move(temp[i++]);
  }

  while (j < n1 + n2) {
    data[k++] = std::move(temp[j++]);
  }
}

void sequential_mergesort(std::vector<std::unique_ptr<Record>> &data,
                          size_t left, size_t right) {
  if (left >= right)
    return;

  // Use insertion sort for small subarrays
  if (right - left < 32) {
    for (size_t i = left + 1; i <= right; ++i) {
      auto key = data[i]->key;
      size_t j = i;

      while (j > left && data[j - 1]->key > key) {
        std::swap(data[j], data[j - 1]);
        j--;
      }
    }
    return;
  }

  size_t mid = left + (right - left) / 2;

  sequential_mergesort(data, left, mid);
  sequential_mergesort(data, mid + 1, right);

  // Skip merge if already sorted
  if (data[mid]->key <= data[mid + 1]->key) {
    return;
  }

  merge(data, left, mid, right);
}

void sequential_mergesort(std::vector<std::unique_ptr<Record>> &data) {
  if (data.size() > 1) {
    sequential_mergesort(data, 0, data.size() - 1);
  }
}

void stl_sort(std::vector<std::unique_ptr<Record>> &data) {
  std::sort(data.begin(), data.end(),
            [](const std::unique_ptr<Record> &a,
               const std::unique_ptr<Record> &b) { return a->key < b->key; });
}
