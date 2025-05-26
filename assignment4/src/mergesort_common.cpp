#include "mergesort_common.h"
#include <algorithm> // For std::min (used elsewhere, but good include hygiene)
#include <cstring>   // For std::memcpy
#include <iostream>  // For std::cerr (error reporting)
#include <memory>    // For std::unique_ptr, std::make_unique
#include <new>       // For std::bad_alloc

// Definition of the threshold for switching to insertion sort.
const size_t INSERTION_SORT_THRESHOLD = 32;

// Sorts a small segment of Record objects in-place using Insertion Sort.
// Chosen for its efficiency on small or nearly-sorted data segments due to
// low overhead compared to merge sort's recursion.
void insertion_sort_records(Record *array_segment_start, size_t count,
                            size_t r_payload_size_bytes) {
  if (count <= 1) {
    // Segments of 0 or 1 elements are inherently sorted.
    return;
  }

  // Use a stack-allocated Record for the key element during insertion.
  // alignas(Record) char temp_record_buffer[sizeof(Record)];
  // Record *key_record = reinterpret_cast<Record *>(temp_record_buffer);
  // Using a direct Record variable is simpler if alignment is not a
  // micro-optimization concern here.
  Record key_record_holder;

  for (size_t i = 1; i < count; ++i) {
    // Store the current element to be inserted.
    copy_record(&key_record_holder, &array_segment_start[i],
                r_payload_size_bytes);

    long j = static_cast<long>(i) - 1;
    // Shift elements in the sorted portion (0 to i-1) that are greater than
    // key_record_holder one position to the right.
    while (j >= 0 && array_segment_start[j].key > key_record_holder.key) {
      copy_record(&array_segment_start[j + 1], &array_segment_start[j],
                  r_payload_size_bytes);
      j--;
    }
    // Insert key_record_holder into its correct sorted position.
    copy_record(&array_segment_start[j + 1], &key_record_holder,
                r_payload_size_bytes);
  }
}

// Merges two sorted adjacent sub-arrays within records_array_base.
// Uses temp_storage_records for the merge and copies result back.
void merge_records(Record *records_array_base,
                   size_t left_start_offset_elements, size_t left_count,
                   size_t right_start_offset_elements, size_t right_count,
                   size_t r_payload_size_bytes,
                   Record *temp_storage_records) { // Changed char* to Record*
  if (left_count == 0 && right_count == 0) {
    return;
  }
  // If left is empty, and if the right part is not already where the output
  // should be, copy it. The output destination is records_array_base +
  // left_start_offset_elements.
  if (left_count == 0) {
    if (right_count > 0 &&
        (left_start_offset_elements != right_start_offset_elements)) {
      Record *dest_ptr = &records_array_base[left_start_offset_elements];
      Record *src_ptr = &records_array_base[right_start_offset_elements];
      for (size_t i = 0; i < right_count; ++i) {
        copy_record(&dest_ptr[i], &src_ptr[i], r_payload_size_bytes);
      }
    }
    return;
  }
  // If right is empty, left part is already in place and sorted.
  if (right_count == 0) {
    return;
  }

  Record *left_ptr = &records_array_base[left_start_offset_elements];
  Record *right_ptr = &records_array_base[right_start_offset_elements];

  size_t i = 0, j = 0,
         k = 0; // Indices for left, right, and temp_storage respectively.

  // Merge elements into temp_storage_records.
  while (i < left_count && j < right_count) {
    if (left_ptr[i].key <= right_ptr[j].key) {
      copy_record(&temp_storage_records[k++], &left_ptr[i++],
                  r_payload_size_bytes);
    } else {
      copy_record(&temp_storage_records[k++], &right_ptr[j++],
                  r_payload_size_bytes);
    }
  }
  // Copy any remaining elements from the left sub-array.
  while (i < left_count) {
    copy_record(&temp_storage_records[k++], &left_ptr[i++],
                r_payload_size_bytes);
  }
  // Copy any remaining elements from the right sub-array.
  while (j < right_count) {
    copy_record(&temp_storage_records[k++], &right_ptr[j++],
                r_payload_size_bytes);
  }

  // Copy the merged result from temp_storage_records back to the original array
  // section.
  Record *destination_in_original_array =
      &records_array_base[left_start_offset_elements];
  for (size_t l = 0; l < (left_count + right_count); ++l) {
    copy_record(&destination_in_original_array[l], &temp_storage_records[l],
                r_payload_size_bytes);
  }
}

// Recursive core of sequential merge sort.
void sequential_merge_sort_recursive(
    Record *array_segment_start, size_t count, size_t r_payload_size_bytes,
    Record *temp_storage_records) { // Changed char* to Record*
  if (count <= 1) {
    // Base case for recursion: segment is already sorted.
    return;
  }
  // Switch to insertion sort for small segments to reduce overhead.
  if (count < INSERTION_SORT_THRESHOLD) {
    insertion_sort_records(array_segment_start, count, r_payload_size_bytes);
    return;
  }

  size_t mid_point = count / 2;
  size_t left_sub_array_count = mid_point;
  size_t right_sub_array_count = count - mid_point;

  // Calculate start pointer for the right sub-segment using array indexing.
  Record *right_sub_array_start = &array_segment_start[left_sub_array_count];

  // Recursively sort the two sub-segments.
  sequential_merge_sort_recursive(array_segment_start, left_sub_array_count,
                                  r_payload_size_bytes, temp_storage_records);
  sequential_merge_sort_recursive(right_sub_array_start, right_sub_array_count,
                                  r_payload_size_bytes, temp_storage_records);

  // Merge the sorted sub-segments. Offsets are relative to array_segment_start.
  // The left part starts at offset 0 relative to array_segment_start.
  // The right part starts at offset left_sub_array_count relative to
  // array_segment_start.
  merge_records(array_segment_start, 0, left_sub_array_count,
                left_sub_array_count, right_sub_array_count,
                r_payload_size_bytes, temp_storage_records);
}

// Public interface for sequential merge sort.
// Manages temporary buffer allocation.
void sequential_merge_sort(Record *records_array, size_t n_elements,
                           size_t r_payload_size_bytes) {
  if (n_elements <= 1) {
    return;
  }
  // Allocate a single temporary buffer of Records.
  std::unique_ptr<Record[]> temp_storage_uptr;
  try {
    temp_storage_uptr = std::make_unique<Record[]>(n_elements);
  } catch (const std::bad_alloc &e) {
    std::cerr << "Error: Failed to allocate temporary storage for sequential "
                 "merge sort ("
              << n_elements << " records). "
              << e.what() // Updated error message
              << std::endl;
    throw; // Critical error, propagate to caller.
  }

  sequential_merge_sort_recursive(
      records_array, n_elements, r_payload_size_bytes, temp_storage_uptr.get());
  // temp_storage_uptr is automatically deallocated when it goes out of scope.
}

// Merges two distinct sorted arrays into a pre-allocated output_array.
// This function was already mostly correct, using array indexing and
// copy_record. No changes needed here based on the current plan, but ensure
// it's consistent.
void merge_two_distinct_arrays(Record *output_array, const Record *left_array,
                               size_t left_count, const Record *right_array,
                               size_t right_count,
                               size_t r_payload_size_bytes) {
  if (!output_array) {
    // This check is important as the caller pre-allocates output_array.
    std::cerr
        << "Error: Output array pointer is null in merge_two_distinct_arrays."
        << std::endl;
    return; // Or throw, depending on error handling policy.
  }
  // It's valid for left_array or right_array to be null if their respective
  // counts are 0.
  if (!left_array && left_count > 0) {
    std::cerr
        << "Error: Left input array pointer is null but left_count is > 0."
        << std::endl;
    return;
  }
  if (!right_array && right_count > 0) {
    std::cerr
        << "Error: Right input array pointer is null but right_count is > 0."
        << std::endl;
    return;
  }

  size_t i = 0; // Index for left_array
  size_t j = 0; // Index for right_array
  size_t k = 0; // Index for output_array

  // Main merge loop, using copy_record for payload-aware copying.
  while (i < left_count && j < right_count) {
    if (left_array[i].key <= right_array[j].key) {
      copy_record(&output_array[k++], &left_array[i++], r_payload_size_bytes);
    } else {
      copy_record(&output_array[k++], &right_array[j++], r_payload_size_bytes);
    }
  }

  // Copy any remaining elements from left_array.
  while (i < left_count) {
    copy_record(&output_array[k++], &left_array[i++], r_payload_size_bytes);
  }

  // Copy any remaining elements from right_array.
  while (j < right_count) {
    copy_record(&output_array[k++], &right_array[j++], r_payload_size_bytes);
  }
}
