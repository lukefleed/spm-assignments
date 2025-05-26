#include "mergesort_hybrid.h"
#include "mergesort_common.h" // For copy_record
#include <algorithm> // For heap operations (std::make_heap, std::pop_heap, std::push_heap, std::sort_heap)
#include <iostream>  // For error reporting if necessary
#include <queue>     // For std::priority_queue alternative
#include <vector>

// Comparator for RecordChunk pointers, used in the min-heap for k-way merge.
// It compares based on the 'key' of the current record each chunk points to.
struct RecordChunkPtrCompare {
  bool operator()(const RecordChunk *a, const RecordChunk *b) const {
    // For a min-heap (e.g., std::priority_queue or when using std::pop_heap to
    // get the smallest), we want 'true' if 'a' should come after 'b' (i.e.,
    // a->key is greater than b->key).
    return a->peek().key > b->peek().key;
  }
};

void sequential_k_way_merge_on_root(
    const Record *concatenated_sorted_chunks_buffer, // Source: all P sorted
                                                     // chunks concatenated
    size_t n_total_elements, // Total number of records across all chunks
    int num_chunks_to_merge, // Number of chunks (P, usually world_size)
    const int *individual_chunk_lengths, // Array of lengths for each chunk
    const int *individual_chunk_displacements, // Array of displacements for
                                               // each chunk in source buffer
    Record *final_output_buffer, // Destination for the final sorted
                                 // N_total_elements
    size_t r_payload_size_bytes) {

  if (n_total_elements == 0) {
    return; // Nothing to merge.
  }

  if (num_chunks_to_merge <= 0) {
    // This case should ideally not happen if n_total_elements > 0.
    // If it does, it implies an inconsistency.
    std::cerr
        << "Error (sequential_k_way_merge_on_root): num_chunks_to_merge is "
        << num_chunks_to_merge << " but n_total_elements is "
        << n_total_elements << ". Cannot proceed." << std::endl;
    // Depending on error handling policy, might throw or fill output with known
    // bad data. For now, just return, output will be undefined/partially
    // filled.
    return;
  }

  // If only one chunk, copy it directly to the output.
  if (num_chunks_to_merge == 1) {
    if (static_cast<size_t>(individual_chunk_lengths[0]) != n_total_elements) {
      std::cerr << "Error (sequential_k_way_merge_on_root): Mismatch in "
                   "elements for single chunk case."
                << " Expected " << n_total_elements << ", got "
                << individual_chunk_lengths[0] << std::endl;
      // Handle error: maybe copy min(n_total_elements, chunk_length)
    }
    // The source data is const, but copy_record needs non-const dest.
    // The actual data pointed to by concatenated_sorted_chunks_buffer is not
    // modified here.
    for (size_t i = 0; i < n_total_elements; ++i) {
      copy_record(
          &final_output_buffer[i],
          &concatenated_sorted_chunks_buffer[individual_chunk_displacements[0] +
                                             i],
          r_payload_size_bytes);
    }
    return;
  }

  // --- K-Way Merge using a Min-Priority Queue ---

  // Vector to hold RecordChunk objects. These objects manage pointers and
  // indices into the source buffer.
  std::vector<RecordChunk> actual_chunks;
  actual_chunks.reserve(num_chunks_to_merge);

  for (int i = 0; i < num_chunks_to_merge; ++i) {
    if (individual_chunk_lengths[i] > 0) {
      // Create a RecordChunk for each non-empty segment.
      // The source buffer is const, but RecordChunk.data is Record*.
      // This is safe as RecordChunk only reads from this data via peek/consume.
      Record *chunk_start_ptr =
          const_cast<Record *>(&concatenated_sorted_chunks_buffer
                                   [individual_chunk_displacements[i]]);
      actual_chunks.emplace_back(
          chunk_start_ptr, static_cast<size_t>(individual_chunk_lengths[i]));
    }
  }

  if (actual_chunks.empty()) { // All provided chunks were empty, though
                               // num_chunks_to_merge > 0
    if (n_total_elements > 0) {
      std::cerr << "Warning (sequential_k_way_merge_on_root): "
                   "num_chunks_to_merge > 0 but all chunks are empty, "
                << "while n_total_elements = " << n_total_elements << std::endl;
    }
    return; // No actual data to merge
  }

  // Use std::priority_queue as a min-heap of RecordChunk pointers.
  // It needs the container type (std::vector<RecordChunk*>) and the custom
  // comparator.
  std::priority_queue<RecordChunk *, std::vector<RecordChunk *>,
                      RecordChunkPtrCompare>
      min_heap;

  // Initialize the heap with the first element from each actual (non-empty)
  // chunk.
  for (size_t i = 0; i < actual_chunks.size(); ++i) {
    min_heap.push(&actual_chunks[i]);
  }

  size_t output_idx = 0;
  while (!min_heap.empty() && output_idx < n_total_elements) {
    // Get the RecordChunk pointer that has the overall smallest current
    // element.
    RecordChunk *smallest_chunk_ptr = min_heap.top();
    min_heap.pop();

    // Copy its current record to the final output buffer. consume() also
    // advances the chunk's internal index.
    copy_record(&final_output_buffer[output_idx++],
                &smallest_chunk_ptr->consume(), r_payload_size_bytes);

    // If the chunk from which the element was taken is not yet exhausted,
    // push it back into the heap (its peek().key will be updated due to
    // consume()).
    if (!smallest_chunk_ptr->is_exhausted()) {
      min_heap.push(smallest_chunk_ptr);
    }
  }

  if (output_idx != n_total_elements) {
    std::cerr << "Error (sequential_k_way_merge_on_root): Merge produced "
              << output_idx << " elements, but " << n_total_elements
              << " were expected." << std::endl;
    // This indicates a potential issue with chunk lengths or the merge logic
    // itself.
  }
}
