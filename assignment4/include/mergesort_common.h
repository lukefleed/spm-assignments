#ifndef MERGESORT_COMMON_H
#define MERGESORT_COMMON_H

#include "record.h" // For Record struct
#include <cstddef>  // For size_t

// Copies a single record, respecting r_payload_size_bytes.
// The actual size of the record in memory is sizeof(Record), but this function
// ensures that only the relevant part of the payload is copied if needed,
// though for fixed-size structs, a direct struct copy is often sufficient and
// simpler. For simplicity and since rpayload is fixed size in Record, direct
// struct copy is fine. If r_payload_size_bytes was meant to dynamically size
// Records, the approach would differ. Given the current Record struct, a simple
// assignment works. This function is provided for conceptual clarity if deeper
// payload handling was intended.
void copy_record_payload_aware(Record *dest, const Record *src,
                               size_t r_payload_size_bytes);

// Simple record copy function (wrapper around copy_record_payload_aware)
void copy_record(Record *dest, const Record *src, size_t r_payload_size_bytes);

// Sorts an array of records in-place using std::sort.
// Relies on a comparator for Record type (based on 'key').
void sequential_sort_records(Record *records, size_t num_records,
                             size_t r_payload_size_bytes);

// Merges two sorted arrays (left_records and right_records) into
// result_records. Assumes result_records has enough space to hold (left_count +
// right_count) elements.
void merge_two_sorted_runs(const Record *left_records, size_t left_count,
                           const Record *right_records, size_t right_count,
                           Record *result_records, size_t r_payload_size_bytes);

#endif // MERGESORT_COMMON_H
