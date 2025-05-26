#ifndef MERGESORT_FF_H
#define MERGESORT_FF_H

#include "record.h" // For struct Record
#include <cstddef>  // For size_t

// Performs a parallel merge sort on a single node using FastFlow.
//
// Parameters:
//   records_array: Pointer to the array of Record objects to be sorted.
//                  The sorting is done in-place.
//   n_elements: The total number of records in the array.
//   r_payload_size_bytes: The actual payload size of each record in bytes.
//   num_threads: The number of FastFlow worker threads to use for
//   parallelization.
void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads);

#endif // MERGESORT_FF_H
