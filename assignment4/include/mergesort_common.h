// include/mergesort_common.h
#ifndef MERGESORT_COMMON_H
#define MERGESORT_COMMON_H

#include "record.h" // Defines Record and MAX_RPAYLOAD_SIZE
#include <cstddef>  // For size_t

/**
 * @brief Copies a Record from source to destination, handling the actual
 * payload size.
 *
 * The key is copied directly. For the payload, 'r_payload_size_bytes' are
 * copied from src->rpayload to dest->rpayload. If 'r_payload_size_bytes' is
 * less than MAX_RPAYLOAD_SIZE, the remaining bytes in dest->rpayload are
 * zero-padded for consistency.
 *
 * @param dest Pointer to the destination Record.
 * @param src Pointer to the source Record.
 * @param r_payload_size_bytes The actual number of payload bytes to copy.
 */
void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes);

#endif // MERGESORT_COMMON_H
