#include "mergesort_ff.h"
#include "mergesort_common.h" // For copy_record, merge_records, RecordKeyCompare
#include "record.h"           // For Record struct definition

#include <algorithm> // For std::sort, an efficient sequential sorting algorithm
#include <cassert>   // For assert
#include <cstring>   // For std::memcpy
#include <ff/dc.hpp> // FastFlow's generic divide-and-conquer component
#include <ff/ff.hpp> // FastFlow core for parallel execution
#include <iostream>  // For std::cerr in case of errors
#include <memory>    // For std::unique_ptr, ensuring RAII for temporary buffers
#include <vector> // For std::vector, used by the divide-and-conquer component

// Defines the size threshold below which array segments are sorted
// sequentially. This is an important tuning parameter for balancing parallelism
// overhead and the efficiency of sequential algorithms on small data sets.
constexpr size_t PARALLEL_SORT_CUTOFF = 131072;

// Global pointers for main and auxiliary buffers for ping-pong strategy
// WARNING: This makes parallel_merge_sort_ff non-reentrant.
static Record *g_main_records_global_ptr = nullptr;
static std::unique_ptr<Record[]> g_aux_records_global_uptr = nullptr;
static Record *g_aux_records_global_ptr_raw =
    nullptr; // Raw pointer for convenience

// Structure representing a unit of work for the parallel sort:
// a specific range (segment) of the array to be processed.
struct DataSegmentToSort {
  size_t offset;             // Offset from the start of the conceptual array.
  size_t length;             // Number of elements in this segment.
  size_t payload_byte_count; // Actual size of the payload in each Record.
  bool source_is_main_array; // True if data for this segment is in
                             // g_main_records_global_ptr
};

// Structure representing the result of processing a DataSegmentToSort.
struct SortedDataSegment {
  size_t offset;             // Offset from the start of the conceptual array.
  size_t element_count;      // Number of elements in this sorted segment.
  size_t payload_byte_count; // Maintained for consistency.
  bool result_is_in_main_array; // True if sorted data is in
                                // g_main_records_global_ptr
};

// Predicate function for the divide-and-conquer framework:
// Returns true if the task (segment) is small enough for sequential processing.
bool is_segment_size_below_cutoff(const DataSegmentToSort &segment) {
  return (segment.length <= PARALLEL_SORT_CUTOFF);
}

// Sequential sorting function for the base cases of the recursion:
// Sorts the provided data segment using std::sort. The sort is in-place
// in the buffer where the source data resides.
void apply_sequential_sort_to_segment(const DataSegmentToSort &segment,
                                      SortedDataSegment &result_marker) {
  if (segment.length > 0) {
    Record *source_ptr =
        (segment.source_is_main_array ? g_main_records_global_ptr
                                      : g_aux_records_global_ptr_raw) +
        segment.offset;
    std::sort(source_ptr, source_ptr + segment.length, RecordKeyCompare());
  }
  result_marker.offset = segment.offset;
  result_marker.element_count = segment.length;
  result_marker.payload_byte_count = segment.payload_byte_count;
  result_marker.result_is_in_main_array = segment.source_is_main_array;
}

// Decomposition function for the divide-and-conquer framework:
// Splits a larger sorting task into two smaller sub-tasks.
void partition_data_segment(const DataSegmentToSort &segment_to_split,
                            std::vector<DataSegmentToSort> &sub_segments) {
  if (segment_to_split.length <= 1) {
    return;
  }

  size_t first_half_length = segment_to_split.length / 2;

  DataSegmentToSort left_sub_segment;
  left_sub_segment.offset = segment_to_split.offset;
  left_sub_segment.length = first_half_length;
  left_sub_segment.payload_byte_count = segment_to_split.payload_byte_count;
  left_sub_segment.source_is_main_array = segment_to_split.source_is_main_array;
  sub_segments.push_back(left_sub_segment);

  DataSegmentToSort right_sub_segment;
  right_sub_segment.offset = segment_to_split.offset + first_half_length;
  right_sub_segment.length = segment_to_split.length - first_half_length;
  right_sub_segment.payload_byte_count = segment_to_split.payload_byte_count;
  right_sub_segment.source_is_main_array =
      segment_to_split.source_is_main_array;
  sub_segments.push_back(right_sub_segment);
}

// Merging function for the divide-and-conquer framework:
// Combines two adjacently sorted data segments into a single, larger sorted
// segment using the ping-pong strategy (merging into the "other" buffer).
void merge_processed_segments(
    std::vector<SortedDataSegment> &sorted_sub_segments,
    SortedDataSegment &final_merged_segment_marker) {
  if (sorted_sub_segments.size() != 2) {
    std::cerr
        << "Error: merge_processed_segments expects 2 sub-segments, received "
        << sorted_sub_segments.size() << ". Merge operation aborted."
        << std::endl;
    if (sorted_sub_segments.empty()) {
      final_merged_segment_marker.offset = 0;
      final_merged_segment_marker.element_count = 0;
      final_merged_segment_marker.payload_byte_count = 0;
      final_merged_segment_marker.result_is_in_main_array = true; // Default
    } else {
      // Fallback, data might be inconsistent or partially merged.
      // Copy properties from the first segment. This is not ideal.
      final_merged_segment_marker = sorted_sub_segments[0];
    }
    return;
  }

  SortedDataSegment &left_segment_res = sorted_sub_segments[0];
  SortedDataSegment &right_segment_res = sorted_sub_segments[1];

  // Assertion: Both sub-segments should be in the same buffer type (main or
  // aux) as they are results from the same level of recursion.
  assert(left_segment_res.result_is_in_main_array ==
         right_segment_res.result_is_in_main_array);

  Record *left_source_ptr = (left_segment_res.result_is_in_main_array
                                 ? g_main_records_global_ptr
                                 : g_aux_records_global_ptr_raw) +
                            left_segment_res.offset;
  Record *right_source_ptr = (right_segment_res.result_is_in_main_array
                                  ? g_main_records_global_ptr
                                  : g_aux_records_global_ptr_raw) +
                             right_segment_res.offset;

  // Determine the target buffer: it's the "other" buffer.
  bool merge_target_is_main_array = !left_segment_res.result_is_in_main_array;
  Record *target_buffer_base = merge_target_is_main_array
                                   ? g_main_records_global_ptr
                                   : g_aux_records_global_ptr_raw;
  // The merged segment will start at the offset of the left sub-segment.
  Record *merge_destination_ptr = target_buffer_base + left_segment_res.offset;

  size_t total_merged_length =
      left_segment_res.element_count + right_segment_res.element_count;
  size_t payload_size = left_segment_res.payload_byte_count;

  if (total_merged_length == 0) {
    final_merged_segment_marker.offset = left_segment_res.offset;
    final_merged_segment_marker.element_count = 0;
    final_merged_segment_marker.payload_byte_count = payload_size;
    final_merged_segment_marker.result_is_in_main_array =
        merge_target_is_main_array;
    return;
  }

  // Perform the merge directly into the target buffer.
  // merge_records does not allocate; it uses the provided destination.
  merge_records(merge_destination_ptr, left_source_ptr,
                left_segment_res.element_count, right_source_ptr,
                right_segment_res.element_count, payload_size);

  // Update the marker for the final merged segment.
  final_merged_segment_marker.offset = left_segment_res.offset;
  final_merged_segment_marker.element_count = total_merged_length;
  final_merged_segment_marker.payload_byte_count = payload_size;
  final_merged_segment_marker.result_is_in_main_array =
      merge_target_is_main_array;
}

// Public interface for performing parallel merge sort using FastFlow.
void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads) {
  if (n_elements <= 1) {
    return;
  }

  // Initialize global buffer pointers
  g_main_records_global_ptr = records_array;
  try {
    g_aux_records_global_uptr = std::make_unique<Record[]>(n_elements);
    g_aux_records_global_ptr_raw = g_aux_records_global_uptr.get();
  } catch (const std::bad_alloc &e) {
    std::cerr << "FATAL ERROR: Failed to allocate auxiliary buffer for merge "
                 "sort (size: "
              << n_elements << " records). " << e.what() << std::endl;
    // Reset pointers in case of error before rethrow or exit
    g_main_records_global_ptr = nullptr;
    g_aux_records_global_uptr.reset();
    g_aux_records_global_ptr_raw = nullptr;
    throw; // Critical failure
  }

  int ff_worker_count = (num_threads > 0) ? num_threads : ff_numCores();
  if (ff_worker_count <= 0) {
    ff_worker_count = 1;
  }

  DataSegmentToSort initial_sorting_task;
  initial_sorting_task.offset = 0;
  initial_sorting_task.length = n_elements;
  initial_sorting_task.payload_byte_count = r_payload_size_bytes;
  initial_sorting_task.source_is_main_array =
      true; // Data starts in the main array

  SortedDataSegment final_operation_marker; // Result will be stored here

  ff::ff_DC<DataSegmentToSort, SortedDataSegment> ff_sorter(
      partition_data_segment,           // divide_f_t
      merge_processed_segments,         // combine_f_t
      apply_sequential_sort_to_segment, // seq_f_t
      is_segment_size_below_cutoff,     // cond_f_t
      initial_sorting_task,             // OperandType op
      final_operation_marker,           // ResultType res
      ff_worker_count                   // numw
  );

  ff_sorter.run_and_wait_end();

  // After sorting, if the final result is in the auxiliary buffer, copy it
  // back.
  if (!final_operation_marker.result_is_in_main_array) {
    if (final_operation_marker.element_count == n_elements &&
        final_operation_marker.offset == 0) {
      // std::cerr << "[DEBUG] Final result is in auxiliary buffer. Copying back
      // to main array." << std::endl; Replace loop of copy_record calls with a
      // single memcpy for performance. This assumes that copying the entire
      // Record struct (including full MAX_RPAYLOAD_SIZE) is acceptable and
      // correct for the user's needs at this stage. If ZERO_UNUSED_PAYLOAD was
      // disabled in copy_record, the behavior for the payload tail (beyond
      // r_payload_size_bytes) might differ slightly, but this is generally
      // faster.
      std::memcpy(g_main_records_global_ptr, g_aux_records_global_ptr_raw,
                  n_elements * sizeof(Record));
    } else {
      // This case should ideally not happen if logic is correct for full sort
      std::cerr << "Warning: Final sorted segment is in auxiliary buffer but "
                   "does not cover the whole array. Data might be inconsistent."
                << std::endl;
    }
  }

  // Clean up global pointers
  g_main_records_global_ptr = nullptr;
  g_aux_records_global_uptr.reset();
  g_aux_records_global_ptr_raw = nullptr;
}
