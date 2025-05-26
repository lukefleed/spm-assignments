#ifndef MERGESORT_COMMON_H
#define MERGESORT_COMMON_H

#include "record.h"
#include <cstddef> // For size_t

// Threshold for switching from merge sort to insertion sort for small
// subarrays. Insertion sort typically has lower overhead for small N.
extern const size_t INSERTION_SORT_THRESHOLD;

// Performs a recursive, sequential merge sort on an array segment of Record
// objects. This is the core divide-and-conquer logic for the sequential
// algorithm. temp_storage_raw_buffer is passed down to avoid repeated
// allocations in recursion.
void sequential_merge_sort_recursive(
    Record *records_array, size_t count, size_t r_payload_size_bytes,
    Record *temp_storage_records); // MODIFIED: char* to Record*

// Main entry point for sequential merge sort.
// Manages the allocation and deallocation of the temporary buffer required by
// the recursive merge sort implementation.
void sequential_merge_sort(Record *records_array, size_t n_elements,
                           size_t r_payload_size_bytes);

// Merges two sorted adjacent sub-arrays within records_array_base.
// The left sub-array starts at left_start_offset_elements.
// The right sub-array starts at right_start_offset_elements.
// Result is placed back into records_array_base starting at
// left_start_offset_elements, using temp_storage_raw_buffer as intermediate
// workspace.
void merge_records(Record *records_array_base,
                   size_t left_start_offset_elements, size_t left_count,
                   size_t right_start_offset_elements, size_t right_count,
                   size_t r_payload_size_bytes,
                   Record *temp_storage_records); // MODIFIED: char* to Record*

// Merges two distinct sorted arrays (left_array and right_array) into a
// single sorted output_array. This function is for out-of-place merges.
// output_array must be pre-allocated to hold (left_count + right_count)
// elements.
void merge_two_distinct_arrays(Record *output_array, const Record *left_array,
                               size_t left_count, const Record *right_array,
                               size_t right_count, size_t r_payload_size_bytes);

#endif // MERGESORT_COMMON_H
