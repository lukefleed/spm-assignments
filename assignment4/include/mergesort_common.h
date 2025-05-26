#ifndef MERGESORT_COMMON_H
#define MERGESORT_COMMON_H

#include "record.h" // For struct Record and get_record_actual_size
#include <vector>   // For std::vector as a temporary buffer in merge

// Performs a sequential merge sort on an array of Record objects.
// This implementation is a standard, recursive merge sort.
//
// Parameters:
//   records_array: Pointer to the beginning of the array segment to be sorted.
//                  The records are manipulated directly in this array.
//   count: The number of Record elements in the segment to be sorted.
//   r_payload_size_bytes: The actual payload size of each record in bytes.
//                         This is crucial for correct memory operations and
//                         comparisons.
//   temp_storage: A pre-allocated temporary buffer used by the merge step.
//                 Its size must be at least 'count' *
//                 get_record_actual_size(r_payload_size_bytes). Passing it
//                 avoids repeated allocations in recursive calls.
void sequential_merge_sort_recursive(Record *records_array, size_t count,
                                     size_t r_payload_size_bytes,
                                     char *temp_storage_raw_buffer);

// Main entry point for sequential merge sort.
// It handles the allocation of the temporary buffer needed for merging.
//
// Parameters:
//   records_array: Pointer to the array of Record objects to be sorted.
//   n_elements: The total number of records in the array.
//   r_payload_size_bytes: The actual payload size of each record.
void sequential_merge_sort(Record *records_array, size_t n_elements,
                           size_t r_payload_size_bytes);

// Merges two sorted sub-arrays into a single sorted array.
// This is a helper function for sequential_merge_sort_recursive.
// Sub-arrays are:
//   Left:  records_array[left_start] ... records_array[left_start + left_count
//   - 1] Right: records_array[right_start] ... records_array[right_start +
//   right_count - 1]
// The merged result is placed back into records_array starting at
// records_array[left_start].
//
// Parameters:
//   records_array: The main array containing the sub-arrays.
//   left_start_idx: Starting index of the left sub-array.
//   left_count: Number of elements in the left sub-array.
//   right_start_idx: Starting index of the right sub-array (immediately after
//   left_count). right_count: Number of elements in the right sub-array.
//   r_payload_size_bytes: Actual payload size for record memory operations.
//   temp_storage_raw_buffer: Raw character buffer for temporary storage during
//   merge.
void merge_records(Record *records_array, size_t left_start_idx,
                   size_t left_count, size_t right_start_idx,
                   size_t right_count, size_t r_payload_size_bytes,
                   char *temp_storage_raw_buffer);

#endif // MERGESORT_COMMON_H
