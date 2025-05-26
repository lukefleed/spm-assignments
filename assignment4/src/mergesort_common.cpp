#include "mergesort_common.h"
#include <algorithm> // For std::min, std::copy (if not using memcpy directly for whole struct)
#include <cstring>   // For memcpy
#include <iostream>  // For error output, if needed
#include <memory>    // For std::unique_ptr (RAII for temporary buffers)
#include <stdexcept> // For potential error handling if needed

// Copies a single Record from src to dest.
void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes) {
  if (!dest || !src) {
    // Handle null pointers if necessary, though typically an assertion
    // or a pre-condition violation. For HPC, we might assume valid pointers.
    return;
  }
  dest->key = src->key;

  // Ensure payload copy does not exceed MAX_RPAYLOAD_SIZE.
  // The actual amount to copy is the minimum of the specified
  // r_payload_size_bytes and the statically defined MAX_RPAYLOAD_SIZE.
  size_t bytes_to_copy =
      std::min(r_payload_size_bytes, static_cast<size_t>(MAX_RPAYLOAD_SIZE));

  if (bytes_to_copy > 0) {
    std::memcpy(dest->rpayload, src->rpayload, bytes_to_copy);
  }

  // Optional: Zero out the rest of the payload buffer in dest if
  // r_payload_size_bytes < MAX_RPAYLOAD_SIZE. This ensures consistent state if
  // the full MAX_RPAYLOAD_SIZE buffer is ever inspected. However, if
  // performance is paramount and only the first r_payload_size_bytes are
  // relevant, this step can be skipped. For correctness in general comparisons,
  // zeroing is safer.
  if (bytes_to_copy < MAX_RPAYLOAD_SIZE) {
    std::memset(dest->rpayload + bytes_to_copy, 0,
                MAX_RPAYLOAD_SIZE - bytes_to_copy);
  }
}

// Merges two sorted arrays of Records (left_array and right_array) into
// dest_array. Assumes dest_array has enough space.
void merge_records(Record *dest_array, Record *left_array, size_t left_len,
                   Record *right_array, size_t right_len,
                   size_t r_payload_size_bytes) {
  size_t i = 0; // Current index in left_array
  size_t j = 0; // Current index in right_array
  size_t k = 0; // Current index in dest_array

  // Traverse both arrays and insert the smaller element into dest_array
  while (i < left_len && j < right_len) {
    if (left_array[i].key <= right_array[j].key) {
      copy_record(&dest_array[k++], &left_array[i++], r_payload_size_bytes);
    } else {
      copy_record(&dest_array[k++], &right_array[j++], r_payload_size_bytes);
    }
  }

  // If there are remaining elements in left_array, copy them
  while (i < left_len) {
    copy_record(&dest_array[k++], &left_array[i++], r_payload_size_bytes);
  }

  // If there are remaining elements in right_array, copy them
  while (j < right_len) {
    copy_record(&dest_array[k++], &right_array[j++], r_payload_size_bytes);
  }
}

// Internal recursive implementation for sequential merge sort.
// This function sorts the segment records_array[0...n_elements-1].
// temp_buffer is used as scratch space for merging.
void sequential_merge_sort_recursive_impl(Record *records_array,
                                          size_t n_elements,
                                          size_t r_payload_size_bytes,
                                          Record *temp_buffer) {
  // Base case: if the array has 1 or 0 elements, it's already sorted.
  if (n_elements <= 1) {
    return;
  }

  size_t mid = n_elements / 2;

  // Recursively sort the two halves.
  // The key insight for temp_buffer usage is that the recursive calls
  // operate on disjoint logical sub-segments of the *original* array,
  // and correspondingly, they can use disjoint segments of the temp_buffer.
  sequential_merge_sort_recursive_impl(records_array, mid, r_payload_size_bytes,
                                       temp_buffer);
  sequential_merge_sort_recursive_impl(records_array + mid, n_elements - mid,
                                       r_payload_size_bytes, temp_buffer + mid);

  // Merge the two sorted halves.
  // For the merge step, we need to copy one of the halves (e.g., the left one)
  // into the temporary buffer to allow merging back into the original
  // records_array segment. We copy records_array[0...mid-1] to
  // temp_buffer[0...mid-1].
  for (size_t idx = 0; idx < mid; ++idx) {
    copy_record(&temp_buffer[idx], &records_array[idx], r_payload_size_bytes);
  }

  // Now merge temp_buffer[0...mid-1] and records_array[mid...n_elements-1]
  // back into records_array[0...n_elements-1].
  merge_records(records_array, temp_buffer, mid, records_array + mid,
                n_elements - mid, r_payload_size_bytes);
}

// Public interface for sequential merge sort.
// Allocates and manages the temporary buffer if one is not provided (though
// typically it should be).
void sequential_merge_sort_recursive(Record *records_array, size_t n_elements,
                                     size_t r_payload_size_bytes,
                                     Record *temp_buffer_param) {
  if (n_elements <= 1) { // Handle 0 or 1 elements (already sorted)
    return;
  }

  // If an external temp_buffer is provided, use it.
  if (temp_buffer_param) {
    sequential_merge_sort_recursive_impl(
        records_array, n_elements, r_payload_size_bytes, temp_buffer_param);
  } else {
    // If no temp_buffer is provided, allocate one internally.
    // This is less ideal for recursive calls if not managed carefully, but for
    // a single top-level call it's acceptable. For performance in deep
    // recursion, pre-allocating is better.
    std::unique_ptr<Record[]> internal_temp_buffer_uptr;
    try {
      internal_temp_buffer_uptr = std::make_unique<Record[]>(n_elements);
    } catch (const std::bad_alloc &e) {
      std::cerr << "FATAL ERROR in sequential_merge_sort_recursive: Failed to "
                   "allocate internal temporary buffer of size "
                << n_elements << ". " << e.what() << std::endl;
      throw; // Propagate critical error
    }
    sequential_merge_sort_recursive_impl(records_array, n_elements,
                                         r_payload_size_bytes,
                                         internal_temp_buffer_uptr.get());
  }
}
