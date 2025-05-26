#ifndef MERGESORT_FF_H
#define MERGESORT_FF_H

#include "record.h" // For Record struct
#include <cstddef>  // For size_t

// Public API function for FastFlow parallel merge sort.
// Sorts 'records_array' of 'n_elements' in-place.
// 'r_payload_size_bytes' specifies the actual size of the payload within each
// Record. 'num_threads' suggests the number of FastFlow worker threads to use.
void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads);

#endif // MERGESORT_FF_H
