// src/mergesort_ff.cpp
#include "mergesort_ff.h"
#include "mergesort_common.h" // For copy_record

#include <algorithm> // For std::sort, std::min
#include <ff/dc.hpp> // For ff_DC
#include <ff/ff.hpp>
#include <memory> // For std::unique_ptr if managing temp buffer manually (not strictly needed with vector)
#include <vector>

// Define a cutoff for switching to sequential sort.
// This value might need tuning based on Record size and system characteristics.
// A common starting point for objects larger than simple integers is a few
// hundred.
#define MERGESORT_FF_CUTOFF 512

// Internal structure to represent a sub-problem for the ff_DC pattern
struct MergeSortTaskData {
  Record *data;        // Pointer to the start of the sub-array
  size_t num_elements; // Number of elements in this sub-array

  // Default constructor is required by ff_DC for the result type
  MergeSortTaskData() : data(nullptr), num_elements(0) {}

  MergeSortTaskData(Record *d, size_t n) : data(d), num_elements(n) {}
};

void parallel_merge_sort_ff(Record *records_array, size_t num_elements,
                            size_t r_payload_size_bytes, int num_threads) {
  if (num_elements == 0) {
    return; // Nothing to sort
  }

  // --- Define Divide and Conquer functions for ff_DC ---

  // Condition function: determines if the problem is small enough for
  // sequential execution
  auto cond_fn = [](const MergeSortTaskData &task) -> bool {
    return task.num_elements <= MERGESORT_FF_CUTOFF;
  };

  // Sequential function: solves the base case
  auto seq_fn = [r_payload_size_bytes](const MergeSortTaskData &task,
                                       MergeSortTaskData &result) {
    // Sort the sub-array in-place using std::sort
    std::sort(task.data, task.data + task.num_elements,
              [](const Record &a, const Record &b) { return a.key < b.key; });
    // Result metadata can point to the same (now sorted) data
    result.data = task.data;
    result.num_elements = task.num_elements;
  };

  // Divide function: splits the problem into sub-problems
  auto divide_fn = [](const MergeSortTaskData &task,
                      std::vector<MergeSortTaskData> &sub_tasks) {
    if (task.num_elements <= 1)
      return; // Should be caught by cond_fn or handled

    size_t mid_point = task.num_elements / 2;

    // Left sub-problem
    sub_tasks.emplace_back(task.data, mid_point);
    // Right sub-problem
    sub_tasks.emplace_back(task.data + mid_point,
                           task.num_elements - mid_point);
  };

  // Combine function: merges results from sub-problems
  auto combine_fn = [r_payload_size_bytes](
                        std::vector<MergeSortTaskData> &sub_results,
                        MergeSortTaskData &result) {
    // Expecting two sub-results for MergeSort
    if (sub_results.size() != 2) {
      // This case should ideally not happen if divide_fn always produces 2
      // sub_tasks and ff_DC processes them. If one sub_task is trivial (e.g. 0
      // elements), it might lead to one result. Handle robustly if necessary.
      // For now, assume two valid sub-results.
      if (sub_results.empty()) { // Nothing to combine
        result.data = nullptr;
        result.num_elements = 0;
        return;
      }
      // If only one result, it's already "combined"
      result = sub_results[0];
      return;
    }

    MergeSortTaskData &left_half = sub_results[0];
    MergeSortTaskData &right_half = sub_results[1];

    // Total elements in the merged range
    size_t total_elements = left_half.num_elements + right_half.num_elements;
    if (total_elements == 0) {
      result.data =
          left_half
              .data; // Or right_half.data, should be same start if contiguous
      result.num_elements = 0;
      return;
    }

    // Create a temporary buffer for merging.
    // Using std::vector for automatic memory management.
    std::vector<Record> temp_buffer(total_elements);

    Record *l_ptr = left_half.data;
    Record *r_ptr = right_half.data;
    Record *const l_end = left_half.data + left_half.num_elements;
    Record *const r_end = right_half.data + right_half.num_elements;

    size_t temp_idx = 0;

    // Standard merge logic
    while (l_ptr < l_end && r_ptr < r_end) {
      if (r_ptr->key <
          l_ptr->key) { // Check right first for stability (though std::sort is
                        // not guaranteed stable here) For mergesort, standard
                        // is (l_ptr->key <= r_ptr->key)
        if (r_ptr->key < l_ptr->key) {
          copy_record(&temp_buffer[temp_idx++], r_ptr++, r_payload_size_bytes);
        } else { // l_ptr->key <= r_ptr->key
          copy_record(&temp_buffer[temp_idx++], l_ptr++, r_payload_size_bytes);
        }
      } else { // l_ptr->key <= r_ptr->key to maintain stability from sub-sorts
               // if they were stable
        copy_record(&temp_buffer[temp_idx++], l_ptr++, r_payload_size_bytes);
      }
    }

    // Copy any remaining elements from the left half
    while (l_ptr < l_end) {
      copy_record(&temp_buffer[temp_idx++], l_ptr++, r_payload_size_bytes);
    }
    // Copy any remaining elements from the right half
    while (r_ptr < r_end) {
      copy_record(&temp_buffer[temp_idx++], r_ptr++, r_payload_size_bytes);
    }

    // Copy sorted data from temp_buffer back to the original array segment
    // The original segment starts at left_half.data
    for (size_t i = 0; i < total_elements; ++i) {
      copy_record(left_half.data + i, &temp_buffer[i], r_payload_size_bytes);
    }

    // The result of the combine operation is the merged range
    result.data = left_half.data; // The merge happens into the start of the
                                  // first sub-problem's data area
    result.num_elements = total_elements;
  };

  // --- Setup and run FastFlow Divide and Conquer ---
  MergeSortTaskData initial_problem(records_array, num_elements);
  MergeSortTaskData
      final_result; // ff_DC will populate this (mostly for metadata)

  // Create the ff_DC object
  ff::ff_DC<MergeSortTaskData, MergeSortTaskData> dac_sorter(
      divide_fn, combine_fn, seq_fn, cond_fn, initial_problem,
      final_result, // This will receive the metadata of the final sorted range
      num_threads);

  // Forcing ff_DC to not create its own farm if we want to embed it,
  // or let it manage its threads. If num_threads is 1, it runs sequentially.
  // If num_threads > 1, it creates an internal farm.
  // dac_sorter.set_scheduling_policy(ff::ff_DC<...>::STATIC); // Or DYNAMIC,
  // GUIDED - if needed

  dac_sorter.run_and_wait_end();
  // The records_array is now sorted in-place.
}
