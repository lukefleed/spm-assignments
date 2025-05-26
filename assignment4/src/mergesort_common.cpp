// src/mergesort_common.cpp
#include "mergesort_common.h"
#include <algorithm> // For std::min
#include <cstring>   // For std::memcpy and std::memset

void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes) {
  if (!dest || !src) {
    // Depending on error handling policy, could throw or log.
    // For HPC, typically avoid branches if inputs are guaranteed valid.
    return;
  }

  dest->key = src->key;

  // Determine the number of payload bytes to actually copy,
  // ensuring it does not exceed the compile-time buffer size.
  size_t bytes_to_copy =
      std::min(r_payload_size_bytes, static_cast<size_t>(MAX_RPAYLOAD_SIZE));

  if (bytes_to_copy >
      0) { // Check to avoid memcpy with zero size, though often benign
    std::memcpy(dest->rpayload, src->rpayload, bytes_to_copy);
  }

  // If the actual payload size used is less than the max buffer size,
  // zero out the remaining part of the destination's payload buffer.
  // This ensures consistent state, crucial for byte-wise comparisons or
  // checksums if they were ever performed on the whole MAX_RPAYLOAD_SIZE
  // region.
  if (bytes_to_copy < MAX_RPAYLOAD_SIZE) {
    std::memset(dest->rpayload + bytes_to_copy, 0,
                MAX_RPAYLOAD_SIZE - bytes_to_copy);
  }
}
