#include "mergesort_common.h"
#include <algorithm> // For std::min (used in other files, but good practice).
#include <cstring>   // For std::memcpy, efficient memory block copying.
#include <iostream>  // For std::cerr, std::endl in case of errors.
#include <new>       // For std::bad_alloc exception.

// Defines the threshold below which recursive MergeSort switches to Insertion
// Sort. Insertion Sort is generally more efficient for very small arrays due to
// lower constant factors and better cache performance, reducing recursive
// overhead. This value is a common heuristic and may be tuned via profiling.
const size_t INSERTION_SORT_THRESHOLD = 32;

// Sorts a small segment of Record objects in-place using Insertion Sort.
// This algorithm is chosen for its efficiency on small or nearly-sorted data
// segments. Parameters:
//   array_segment_start: Pointer to the beginning of the segment to sort.
//   count: Number of Record elements in the segment.
//   r_payload_size_bytes: Actual size of the payload for each Record, used for
//   correct memory copying.
void insertion_sort_records(Record *array_segment_start, size_t count,
                            size_t r_payload_size_bytes) {
  if (count <= 1) {
    // A segment of 0 or 1 elements is inherently sorted.
    return;
  }

  size_t record_mem_size = get_record_actual_size(r_payload_size_bytes);
  // Allocate a temporary buffer on the stack to hold one Record during swaps.
  // Using a fixed max size for the stack buffer to avoid VLA and dynamic
  // allocation here. The actual amount copied will be record_mem_size.
  char temp_record_buffer[sizeof(unsigned long) + MAX_RPAYLOAD_SIZE];
  Record *key_record = reinterpret_cast<Record *>(
      temp_record_buffer); // Typed pointer for easier field access.

  char *base_char_ptr = reinterpret_cast<char *>(array_segment_start);

  // Standard Insertion Sort algorithm:
  // Iterate from the second element to the end of the segment.
  for (size_t i = 1; i < count; ++i) {
    // current_element_ptr points to the element to be inserted into the sorted
    // portion.
    char *current_element_ptr = base_char_ptr + i * record_mem_size;
    // Copy the element to be inserted into temporary storage.
    std::memcpy(key_record, current_element_ptr, record_mem_size);

    // Iterate backwards through the sorted portion (elements 0 to i-1).
    long j =
        static_cast<long>(i) -
        1; // Use signed type for j to correctly handle loop termination j < 0.

    // Shift elements in the sorted portion that are greater than
    // key_record->key one position to the right to make space for key_record.
    while (j >= 0) {
      Record *compare_record =
          reinterpret_cast<Record *>(base_char_ptr + j * record_mem_size);
      if (compare_record->key > key_record->key) {
        // destination: base_char_ptr + (j + 1) * record_mem_size
        // source:      base_char_ptr + j * record_mem_size (which is
        // compare_record)
        std::memcpy(base_char_ptr + (j + 1) * record_mem_size, compare_record,
                    record_mem_size);
        j--;
      } else {
        // Element compare_record->key is less than or equal to key_record->key,
        // so key_record belongs after compare_record.
        break;
      }
    }
    // Insert key_record into its correct sorted position.
    // The position is (j + 1) because j was decremented one last time if the
    // loop condition held, or j points to the element just before the insertion
    // point.
    std::memcpy(base_char_ptr + (j + 1) * record_mem_size, key_record,
                record_mem_size);
  }
}

// Merges two pre-sorted adjacent segments of Record objects.
// The merged result is written back into the location of the first segment,
// expanding to cover both. This function assumes `records_array_base` points to
// a contiguous memory block where the left and right segments reside, and the
// merge happens into `temp_storage_raw_buffer`, then copied back. Parameters:
//   records_array_base: Base pointer of the original full array. Merge
//   operations are relative to this. left_start_offset_elements: Element offset
//   of the left sub-array from records_array_base. left_count: Number of
//   elements in the left sub-array. right_start_offset_elements: Element offset
//   of the right sub-array from records_array_base. right_count: Number of
//   elements in the right sub-array. r_payload_size_bytes: Actual payload size
//   for correct record memory operations. temp_storage_raw_buffer: Raw
//   character buffer for temporary storage during the merge. Must be
//                            large enough to hold (left_count + right_count)
//                            records.
void merge_records(Record *records_array_base,
                   size_t left_start_offset_elements, size_t left_count,
                   size_t right_start_offset_elements, size_t right_count,
                   size_t r_payload_size_bytes, char *temp_storage_raw_buffer) {

  // If either sub-array is empty, there is no need to merge.
  if (left_count == 0 || right_count == 0) {
    return;
  }

  size_t record_mem_size = get_record_actual_size(r_payload_size_bytes);

  // Calculate direct pointers to the start of the left and right sub-arrays.
  char *left_segment_ptr = reinterpret_cast<char *>(records_array_base) +
                           left_start_offset_elements * record_mem_size;
  char *right_segment_ptr = reinterpret_cast<char *>(records_array_base) +
                            right_start_offset_elements * record_mem_size;

  // Iterators for traversing left, right, and temporary merged arrays.
  size_t i = 0; // Current index in the left sub-array.
  size_t j = 0; // Current index in the right sub-array.
  size_t k =
      0; // Current index in the temp_storage_raw_buffer (as element count).

  // Main merge loop: compare elements from left and right sub-arrays
  // and copy the smaller one to the temporary buffer.
  while (i < left_count && j < right_count) {
    Record *rec_left =
        reinterpret_cast<Record *>(left_segment_ptr + i * record_mem_size);
    Record *rec_right =
        reinterpret_cast<Record *>(right_segment_ptr + j * record_mem_size);

    if (rec_left->key <= rec_right->key) {
      std::memcpy(temp_storage_raw_buffer + k * record_mem_size, rec_left,
                  record_mem_size);
      i++;
    } else {
      std::memcpy(temp_storage_raw_buffer + k * record_mem_size, rec_right,
                  record_mem_size);
      j++;
    }
    k++;
  }

  // If elements remain in the left sub-array, copy them to the temporary
  // buffer.
  while (i < left_count) {
    std::memcpy(temp_storage_raw_buffer + k * record_mem_size,
                left_segment_ptr + i * record_mem_size, record_mem_size);
    i++;
    k++;
  }

  // If elements remain in the right sub-array, copy them to the temporary
  // buffer.
  while (j < right_count) {
    std::memcpy(temp_storage_raw_buffer + k * record_mem_size,
                right_segment_ptr + j * record_mem_size, record_mem_size);
    j++;
    k++;
  }

  // Copy the fully merged segment from temporary storage back to the original
  // array. The destination in the original array starts at the beginning of the
  // left sub-array's original position.
  char *destination_in_original_array =
      reinterpret_cast<char *>(records_array_base) +
      left_start_offset_elements * record_mem_size;
  std::memcpy(destination_in_original_array, temp_storage_raw_buffer,
              (left_count + right_count) * record_mem_size);
}

// Recursive core of the sequential MergeSort algorithm.
// Sorts the segment of records starting at `array_segment_start` containing
// `count` elements. Uses `temp_storage_raw_buffer` for merge operations to
// avoid repeated allocations.
void sequential_merge_sort_recursive(Record *array_segment_start, size_t count,
                                     size_t r_payload_size_bytes,
                                     char *temp_storage_raw_buffer) {
  if (count <= 1) {
    // Base case for recursion: a segment of 0 or 1 elements is already sorted.
    return;
  }

  // Optimization: Switch to Insertion Sort for small segments.
  if (count < INSERTION_SORT_THRESHOLD) {
    insertion_sort_records(array_segment_start, count, r_payload_size_bytes);
    return;
  }

  // Divide: Calculate midpoint and sizes of left and right sub-segments.
  size_t mid_point = count / 2;
  size_t left_sub_array_count = mid_point;
  size_t right_sub_array_count = count - mid_point;

  // Determine the starting pointer for the right sub-segment.
  Record *right_sub_array_start = reinterpret_cast<Record *>(
      reinterpret_cast<char *>(array_segment_start) +
      left_sub_array_count * get_record_actual_size(r_payload_size_bytes));

  // Conquer: Recursively sort the two sub-segments.
  sequential_merge_sort_recursive(array_segment_start, left_sub_array_count,
                                  r_payload_size_bytes,
                                  temp_storage_raw_buffer);
  sequential_merge_sort_recursive(right_sub_array_start, right_sub_array_count,
                                  r_payload_size_bytes,
                                  temp_storage_raw_buffer);

  // Combine: Merge the two sorted sub-segments.
  // The offsets for `merge_records` are relative to `array_segment_start`
  // because that's the "base" for this level of recursion's merge operation.
  merge_records(array_segment_start, /*left_start_offset_elements=*/0,
                left_sub_array_count,
                /*right_start_offset_elements=*/left_sub_array_count,
                right_sub_array_count, r_payload_size_bytes,
                temp_storage_raw_buffer);
}

// Public interface for sequential merge sort.
// Allocates a single temporary buffer for the entire sort operation to optimize
// performance by avoiding frequent memory allocations/deallocations during
// recursion.
void sequential_merge_sort(Record *records_array, size_t n_elements,
                           size_t r_payload_size_bytes) {
  if (n_elements <= 1) {
    // An array with 0 or 1 elements does not need sorting.
    return;
  }

  size_t record_mem_size = get_record_actual_size(r_payload_size_bytes);
  char *temp_storage_raw_buffer = nullptr;
  try {
    // Allocate temporary storage for the merge phase.
    // This buffer needs to be large enough to hold all elements being sorted.
    temp_storage_raw_buffer = new char[n_elements * record_mem_size];
  } catch (const std::bad_alloc &e) {
    // Handle memory allocation failure criticaly.
    std::cerr << "Error: Failed to allocate temporary storage for sequential "
                 "merge sort ("
              << n_elements * record_mem_size << " bytes). " << e.what()
              << std::endl;
    throw; // Re-throw to allow the caller to handle this critical error.
  }

  sequential_merge_sort_recursive(
      records_array, n_elements, r_payload_size_bytes, temp_storage_raw_buffer);

  // Deallocate the temporary storage.
  delete[] temp_storage_raw_buffer;
}
