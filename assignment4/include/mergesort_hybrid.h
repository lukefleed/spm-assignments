#ifndef MERGESORT_HYBRID_H
#define MERGESORT_HYBRID_H

#include "record.h" // For Record struct
#include <cstddef>  // For size_t
#include <vector>   // For std::vector (will be used for RecordChunk management)

// Represents a chunk of data, typically sorted, to be merged.
// This struct will be used by the k-way merge logic on the root process.
struct RecordChunk {
  Record
      *data; // Pointer to the start of the chunk's data within a larger buffer
  size_t num_elements; // Number of elements in this specific chunk
  size_t current_idx;  // Current index for merging (tracks progress within this
                       // chunk)

  // Constructor
  RecordChunk(Record *d, size_t n) : data(d), num_elements(n), current_idx(0) {}

  // Returns true if all elements from this chunk have been processed during
  // merge.
  bool is_exhausted() const { return current_idx >= num_elements; }

  // Gets the current record (key and payload) without advancing the index.
  // Assumes !is_exhausted().
  const Record &peek() const { return data[current_idx]; }

  // Gets the current record and advances the index to the next element.
  // Assumes !is_exhausted().
  const Record &consume() { return data[current_idx++]; }
};

// Performs a k-way merge of 'num_chunks_to_merge' sorted Record arrays.
// These chunks are assumed to be located contiguously within
// 'concatenated_sorted_chunks_buffer'. 'individual_chunk_lengths': An array
// where individual_chunk_lengths[i] is the number of elements
//                             in the i-th chunk.
// 'individual_chunk_displacements': An array where
// individual_chunk_displacements[i] is the offset (in number of Records)
//                                  from the beginning of
//                                  'concatenated_sorted_chunks_buffer' to the
//                                  start of the i-th chunk.
// 'final_output_buffer': The destination buffer where the fully sorted result
// of N_total_elements will be written.
//                        Must be pre-allocated to hold N_total_elements.
//
// This function is intended to be called by the root MPI process (rank 0) after
// gathering all locally sorted chunks from MPI processes.
void sequential_k_way_merge_on_root(
    const Record *concatenated_sorted_chunks_buffer, // Source buffer holding
                                                     // all P sorted chunks
    size_t n_total_elements, // Total number of records across all chunks
    int num_chunks_to_merge, // Number of chunks to merge (P, world_size)
    const int *individual_chunk_lengths, // Lengths of each of the P chunks
    const int *individual_chunk_displacements, // Displacements to the start of
                                               // each chunk in source buffer
    Record *final_output_buffer, // Destination for the final sorted
                                 // N_total_elements
    size_t r_payload_size_bytes  // Actual payload size to use for copy_record
);

#endif // MERGESORT_HYBRID_H
