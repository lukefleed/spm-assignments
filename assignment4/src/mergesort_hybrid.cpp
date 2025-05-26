#include "mergesort_hybrid.h"
#include "mergesort_common.h" // For copy_record
#include <algorithm> // For heap operations (std::make_heap, std::pop_heap, etc.)
#include <iostream>  // For error reporting if necessary
#include <vector>

// Comparator for RecordChunk pointers... (come prima)
struct RecordChunkPtrCompare {
  bool operator()(const RecordChunk *a, const RecordChunk *b) const {
    return a->peek().key > b->peek().key;
  }
};

void sequential_k_way_merge_on_root(
    const Record *concatenated_sorted_chunks_buffer, size_t n_total_elements,
    int num_chunks_to_merge, const int *individual_chunk_lengths,
    const int *individual_chunk_displacements, Record *final_output_buffer,
    size_t r_payload_size_bytes) {

  // ... (parte iniziale della funzione come prima) ...

  if (n_total_elements == 0) {
    return;
  }

  if (num_chunks_to_merge == 0) {
    std::cerr << "Warning (sequential_k_way_merge_on_root): "
                 "num_chunks_to_merge is 0, but n_total_elements is "
              << n_total_elements << ". No operation performed." << std::endl;
    return;
  }

  if (num_chunks_to_merge == 1) {
    if (static_cast<size_t>(individual_chunk_lengths[0]) != n_total_elements) {
      std::cerr << "Error (sequential_k_way_merge_on_root): Mismatch in "
                   "elements for single chunk case."
                << std::endl;
    }
    for (size_t i = 0; i < n_total_elements; ++i) {
      copy_record(
          &final_output_buffer[i],
          &concatenated_sorted_chunks_buffer[individual_chunk_displacements[0] +
                                             i],
          r_payload_size_bytes);
    }
    return;
  }

  // TODO: Implement full k-way merge... (commento come prima)

  // Temporary message for stub:
  if (num_chunks_to_merge > 1) { // Questo messaggio verrà ora stampato sempre
                                 // se num_chunks_to_merge > 1
    std::cerr
        << "Warning (sequential_k_way_merge_on_root): Not fully implemented "
           "yet for >1 chunks. Output will be incorrect for this case."
        << std::endl;
  }

  // Fallback for >1 chunks until implemented: copy first chunk if exists, to
  // avoid segfaults in testing main logic. THIS IS NOT CORRECT FOR MERGING,
  // JUST A STUB to allow compilation and basic run of main_hybrid.
  if (num_chunks_to_merge > 1 && individual_chunk_lengths[0] > 0 &&
      n_total_elements > 0) {
    size_t count_to_copy = std::min(
        n_total_elements, static_cast<size_t>(individual_chunk_lengths[0]));
    for (size_t i = 0; i < count_to_copy; ++i) {
      copy_record(
          &final_output_buffer[i],
          &concatenated_sorted_chunks_buffer[individual_chunk_displacements[0] +
                                             i],
          r_payload_size_bytes);
    }
  }
}
