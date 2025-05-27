#include "mergesort_common.h"
#include <algorithm> // For std::sort, std::min
#include <cstring>   // For memcpy

// Custom comparator for Record struct, used by std::sort
struct RecordComparator {
  bool operator()(const Record &a, const Record &b) const {
    return a.key < b.key;
  }
};

// For fixed-size Record, direct assignment is efficient and correct.
// This function is kept if specific payload handling becomes necessary later.
void copy_record_payload_aware(Record *dest, const Record *src,
                               size_t r_payload_size_bytes) {
  // dest->key = src->key;
  // memcpy(dest->rpayload, src->rpayload, std::min(r_payload_size_bytes,
  // MAX_RPAYLOAD_SIZE)); For the given Record struct, a simple struct copy is
  // most straightforward:
  *dest = *src;
  // Suppress unused parameter warning if r_payload_size_bytes is not strictly
  // needed for copy
  (void)r_payload_size_bytes;
}

// Simple wrapper function for copy_record
void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes) {
  copy_record_payload_aware(dest, src, r_payload_size_bytes);
}

void sequential_sort_records(Record *records, size_t num_records,
                             size_t r_payload_size_bytes) {
  if (num_records == 0)
    return;
  // r_payload_size_bytes is part of the Record's state but doesn't change sort
  // logic here as comparison is only on 'key'.
  (void)r_payload_size_bytes; // Suppress unused parameter warning
  std::sort(records, records + num_records, RecordComparator());
}

void merge_two_sorted_runs(const Record *left_records, size_t left_count,
                           const Record *right_records, size_t right_count,
                           Record *result_records,
                           size_t r_payload_size_bytes) {

  (void)r_payload_size_bytes; // Suppress unused parameter, Record copy handles
                              // payload.

  size_t i = 0; // Index for left_records
  size_t j = 0; // Index for right_records
  size_t k = 0; // Index for result_records

  while (i < left_count && j < right_count) {
    if (left_records[i].key <= right_records[j].key) {
      // copy_record_payload_aware(&result_records[k++], &left_records[i++],
      // r_payload_size_bytes);
      result_records[k++] = left_records[i++];
    } else {
      // copy_record_payload_aware(&result_records[k++], &right_records[j++],
      // r_payload_size_bytes);
      result_records[k++] = right_records[j++];
    }
  }

  // Copy remaining elements from left_records, if any
  while (i < left_count) {
    // copy_record_payload_aware(&result_records[k++], &left_records[i++],
    // r_payload_size_bytes);
    result_records[k++] = left_records[i++];
  }

  // Copy remaining elements from right_records, if any
  while (j < right_count) {
    // copy_record_payload_aware(&result_records[k++], &right_records[j++],
    // r_payload_size_bytes);
    result_records[k++] = right_records[j++];
  }
}
