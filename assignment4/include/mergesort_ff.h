// include/mergesort_ff.h
#ifndef MERGESORT_FF_H
#define MERGESORT_FF_H

#include "record.h" // Defines Record
#include <cstddef>  // For size_t

/**
 * @brief Sorts an array of Record objects in parallel using FastFlow's
 *        Divide and Conquer pattern.
 *
 * The sorting is performed in-place on the records_array based on Record::key.
 *
 * @param records_array Pointer to the array of Record objects to be sorted.
 * @param num_elements The number of elements in records_array.
 * @param r_payload_size_bytes The actual size of the payload in each Record,
 *                             used by copy_record.
 * @param num_threads The number of FastFlow worker threads to use.
 */
void parallel_merge_sort_ff(Record *records_array, size_t num_elements,
                            size_t r_payload_size_bytes, int num_threads);

#endif // MERGESORT_FF_H
