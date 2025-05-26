#ifndef MERGESORT_COMMON_H
#define MERGESORT_COMMON_H

#include "record.h" // Ensures Record struct is defined
#include <cstddef>  // For size_t

// Copies a single Record.
// Assumes dest and src point to valid memory locations.
void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes);

// Merges two sorted arrays of Records (left_array and right_array) into
// dest_array. Assumes dest_array has enough space to hold all elements from
// both.
void merge_records(Record *dest_array, Record *left_array, size_t left_len,
                   Record *right_array, size_t right_len,
                   size_t r_payload_size_bytes);

// Sequentially sorts an array of Records using a recursive merge sort.
// Requires a temporary buffer of the same size as records_array for merging.
void sequential_merge_sort_recursive(Record *records_array, size_t n_elements,
                                     size_t r_payload_size_bytes,
                                     Record *temp_buffer);

// Comparator struct for sorting Records based on their keys.
// This is suitable for use with std::sort or other standard algorithms.
struct RecordKeyCompare {
  bool operator()(const Record &a, const Record &b) const {
    return a.key < b.key;
  }
};

#endif // MERGESORT_COMMON_H
