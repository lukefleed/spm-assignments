#ifndef RECORD_H
#define RECORD_H

#include <cstddef> // For size_t
#include <cstring> // For memcpy

// Defines the maximum payload size that the struct can nominally hold.
// The actual payload size used in memory operations will be determined at
// runtime. This value should be at least as large as the maximum RPAYLOAD
// specified in tests (e.g., 256).
#define MAX_RPAYLOAD_SIZE 256

// Represents a single record to be sorted.
// The actual size of a Record instance in memory will depend on the
// runtime R_payload_size, specifically: sizeof(unsigned long) + R_payload_size.
struct Record {
  unsigned long key;                // The value used for sorting.
  char rpayload[MAX_RPAYLOAD_SIZE]; // Buffer for the payload. Only the first
                                    // 'R_payload_size' bytes are used.
};

// Comparison function for sorting records based on their keys.
// Used by std::sort or other sorting algorithms that require a comparator.
inline bool compareRecords(const Record &a, const Record &b) {
  return a.key < b.key;
}

// Helper function to calculate the actual size of a Record in bytes,
// given a specific payload size. This is crucial for memory allocation
// and pointer arithmetic.
inline size_t get_record_actual_size(size_t r_payload_size_bytes) {
  // Ensures that if r_payload_size_bytes is 0, we still account for the key.
  // The struct layout in memory is key followed by payload.
  return sizeof(unsigned long) + r_payload_size_bytes;
}

// Copies data from one Record (src) to another (dest), considering the actual
// payload size. It's important to use this instead of a direct struct
// assignment if R_payload_size is less than MAX_RPAYLOAD_SIZE to avoid copying
// uninitialized/garbage data from the rpayload buffer.
inline void copy_record(Record *dest, const Record *src,
                        size_t r_payload_size_bytes) {
  dest->key = src->key;
  // Only copy the relevant part of the payload.
  std::memcpy(dest->rpayload, src->rpayload, r_payload_size_bytes);
}

#endif // RECORD_H
