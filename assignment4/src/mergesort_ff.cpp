// src/mergesort_ff.cpp
#include "mergesort_ff.h"
#include "mergesort_common.h"

#include <ff/ff.hpp>

#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::log2, std::ceil
#include <iostream>  // For std::cerr for error reporting
#include <memory>    // For std::unique_ptr, std::move
#include <new>       // For std::bad_alloc
#include <vector>

// --- Task definition for initial parallel sorting of chunks ---
struct SortChunkTask {
  Record *data_ptr;                 // Pointer to the data segment to sort
  size_t count;                     // Number of records in this segment
  size_t r_payload_size_bytes_task; // Actual payload size for these records
  Record *temp_buffer_for_worker;   // Pre-allocated buffer (Record array)

  SortChunkTask(Record *ptr, size_t n, size_t r_size,
                Record *temp_buf) // Changed char* to Record*
      : data_ptr(ptr), count(n), r_payload_size_bytes_task(r_size),
        temp_buffer_for_worker(temp_buf) {}
};

// --- FastFlow worker for the initial sorting farm ---
// Each worker sorts one SortChunkTask.
class FFInitialSortWorker : public ff::ff_node_t<SortChunkTask, SortChunkTask> {
public:
  FFInitialSortWorker() = default;

  SortChunkTask *svc(SortChunkTask *task) override {
    if (!task || !task->data_ptr || task->count == 0) {
      // Handles EOS from emitter or invalid tasks.
      // Signals to the farm that this worker is ready for more tasks.
      return reinterpret_cast<SortChunkTask *>(ff::FF_GO_ON);
    }
    // Perform in-place sort on the assigned data segment.
    sequential_merge_sort_recursive(
        task->data_ptr, task->count, task->r_payload_size_bytes_task,
        task->temp_buffer_for_worker); // MODIFIED: Removed
                                       // reinterpret_cast<char *>

    // Signal completion of this task; worker is ready for the next.
    return reinterpret_cast<SortChunkTask *>(ff::FF_GO_ON);
  }
  ~FFInitialSortWorker() override = default;
};

// --- FastFlow emitter for the initial sorting farm ---
// Generates SortChunkTask items for the farm workers.
class TaskEmitter : public ff::ff_node_t<SortChunkTask> {
private:
  std::vector<SortChunkTask>
      &task_descriptors_ref; // Holds all tasks to be emitted.
  size_t next_task_idx;      // Tracks the next task to be dispatched.
  bool eos_sent; // Ensures End-Of-Stream (EOS) is emitted only once.

public:
  TaskEmitter(std::vector<SortChunkTask> &tasks)
      : task_descriptors_ref(tasks), next_task_idx(0), eos_sent(false) {}

  SortChunkTask *svc(SortChunkTask * /*task_ignored*/) override {
    if (next_task_idx < task_descriptors_ref.size()) {
      // Dispatch next available task.
      return &task_descriptors_ref[next_task_idx++];
    }
    if (!eos_sent) {
      // All tasks have been dispatched; send EOS to signal workers to
      // terminate.
      eos_sent = true;
      return reinterpret_cast<SortChunkTask *>(EOS);
    }
    // After EOS, return nullptr to indicate no more tasks from this emitter.
    return nullptr;
  }
  ~TaskEmitter() override = default;
};

// --- Task definition for the parallel merge phase ---
// Represents a sorted chunk of data.
struct MergeTask {
  Record *data_ptr; // Pointer to the start of the sorted data chunk
  size_t count;     // Number of records in this chunk
  size_t r_payload_size_bytes_task; // Actual payload size for records
  bool owns_data; // True if this task is responsible for deallocating data_ptr

  // Constructor for initial chunks (data is part of the main array, not owned
  // by MergeTask) or for tasks where data ownership is managed externally.
  MergeTask(Record *ptr, size_t n, size_t r_size, bool owns = false)
      : data_ptr(ptr), count(n), r_payload_size_bytes_task(r_size),
        owns_data(owns) {}

  // Destructor: frees data_ptr if this task owns the data.
  // This is crucial for intermediate chunks created during merges.
  ~MergeTask() {
    if (owns_data && data_ptr) {
      delete[] data_ptr; // Assumes memory was allocated with new Record[].
    }
  }

  // Rule of 5/3: Disable copy operations to prevent shallow copies and double
  // frees.
  MergeTask(const MergeTask &) = delete;
  MergeTask &operator=(const MergeTask &) = delete;

  // Move constructor: transfers ownership of data_ptr.
  MergeTask(MergeTask &&other) noexcept
      : data_ptr(other.data_ptr), count(other.count),
        r_payload_size_bytes_task(other.r_payload_size_bytes_task),
        owns_data(other.owns_data) {
    other.data_ptr = nullptr; // Source task no longer owns the data.
    other.owns_data = false;
  }

  // Move assignment operator: transfers ownership of data_ptr.
  MergeTask &operator=(MergeTask &&other) noexcept {
    if (this != &other) {
      // Release current resources if owned.
      if (owns_data && data_ptr) {
        delete[] data_ptr;
      }
      // Pilfer resources from other.
      data_ptr = other.data_ptr;
      count = other.count;
      r_payload_size_bytes_task = other.r_payload_size_bytes_task;
      owns_data = other.owns_data;

      // Leave other in a valid, destructible state.
      other.data_ptr = nullptr;
      other.owns_data = false;
    }
    return *this;
  }
};

// --- FastFlow emitter for the merge farm (emits pairs of MergeTasks) ---
struct MergeFarmEmitter : ff::ff_node_t<std::pair<MergeTask, MergeTask>> {
  std::vector<MergeTask>
      &chunks_to_merge_source; // Reference to the vector of chunks for current
                               // merge level.
  size_t next_pair_start_idx;  // Tracks the starting index for the next pair.
  bool eos_sent;

  MergeFarmEmitter(std::vector<MergeTask> &chunks)
      : chunks_to_merge_source(chunks), next_pair_start_idx(0),
        eos_sent(false) {}

  std::pair<MergeTask, MergeTask> *
  svc(std::pair<MergeTask, MergeTask> *) override {
    // Check if there are at least two chunks left to form a pair.
    if (next_pair_start_idx + 1 < chunks_to_merge_source.size()) {
      // Create a new pair, moving MergeTasks from the source vector to ensure
      // proper ownership transfer and avoid copying large data structures.
      auto *task_pair = new std::pair<MergeTask, MergeTask>(
          std::move(chunks_to_merge_source[next_pair_start_idx]),
          std::move(chunks_to_merge_source[next_pair_start_idx + 1]));
      next_pair_start_idx += 2; // Advance index by 2 for the next pair.
      return task_pair;
    }
    if (!eos_sent) {
      eos_sent = true;
      // Signal End-Of-Stream for this level of merging.
      return reinterpret_cast<std::pair<MergeTask, MergeTask> *>(EOS);
    }
    return nullptr; // No more pairs to emit.
  }
  ~MergeFarmEmitter() override = default;
};

// --- FastFlow worker for the merge farm (merges one pair of MergeTasks) ---
struct MergeFarmWorker
    : ff::ff_node_t<std::pair<MergeTask, MergeTask>, MergeTask> {
  MergeFarmWorker() = default;

  MergeTask *svc(std::pair<MergeTask, MergeTask> *task_pair_ptr) override {
    if (!task_pair_ptr) { // Should only happen on farm shutdown or error.
      return nullptr;
    }

    // Take ownership of the MergeTasks from the input pair.
    MergeTask task1 = std::move(task_pair_ptr->first);
    MergeTask task2 = std::move(task_pair_ptr->second);
    delete task_pair_ptr; // Delete the dynamically allocated pair object
                          // itself.

    size_t merged_count = task1.count + task2.count;
    // If both input tasks are empty, return an empty owned task.
    if (merged_count == 0) {
      return new MergeTask(nullptr, 0, task1.r_payload_size_bytes_task, true);
    }
    // Assume payload size is consistent across tasks from the same sort
    // operation.
    size_t r_payload = task1.r_payload_size_bytes_task;

    Record *merged_data_ptr = nullptr;
    try {
      // Allocate memory for the merged result. This memory will be owned by the
      // output MergeTask.
      merged_data_ptr = new Record[merged_count];
    } catch (const std::bad_alloc &e) {
      std::cerr
          << "MergeFarmWorker: Failed to allocate memory for merged chunk of "
          << merged_count << " records: " << e.what() << std::endl;
      // Destructors of task1 and task2 will handle cleanup of their data if
      // they owned it.
      return nullptr; // Propagate error by returning nullptr. Farm collector
                      // should handle this.
    }

    // Perform the 2-way merge using the common utility function.
    merge_two_distinct_arrays(merged_data_ptr,             // Output buffer
                              task1.data_ptr, task1.count, // Left input chunk
                              task2.data_ptr, task2.count, // Right input chunk
                              r_payload // Payload size for copying
    );

    // task1 and task2 go out of scope here. If they owned data, their
    // destructors will free it. The new MergeTask owns the merged_data_ptr.
    return new MergeTask(merged_data_ptr, merged_count, r_payload, true);
  }
  ~MergeFarmWorker() override = default;
};

// --- FastFlow collector for the merge farm ---
// Collects MergeTask results from workers.
struct MergeFarmCollector
    : ff::ff_node_t<MergeTask> { // Input-only node type for farm collector
  std::vector<MergeTask>
      &collected_chunks_ref; // Stores collected tasks for the next merge level.

  MergeFarmCollector(std::vector<MergeTask> &collected_chunks)
      : collected_chunks_ref(collected_chunks) {}

  // The input task_ptr is technically const void* from the farm's perspective,
  // but we cast it to MergeTask* as that's what MergeFarmWorker produces.
  MergeTask *svc(MergeTask *task_ptr) override {
    if (task_ptr) { // A valid merged task (or an error indicator from worker)
      if (task_ptr->data_ptr == nullptr && task_ptr->count > 0) {
        // This case indicates a worker failed to produce valid data but
        // returned a task.
        std::cerr << "MergeFarmCollector: Received a task indicating worker "
                     "error (nullptr data with count > 0)."
                  << std::endl;
        delete task_ptr; // Delete the error-indicating task wrapper.
      } else if (task_ptr->data_ptr != nullptr || task_ptr->count == 0) {
        // Valid task (possibly empty if inputs were empty). Move it to the
        // collection.
        collected_chunks_ref.emplace_back(std::move(*task_ptr));
        delete task_ptr; // Delete the task wrapper object, ownership of data
                         // moved.
      }
    } else {
      // Receiving a nullptr here could mean a worker failed critically and
      // returned nullptr.
      std::cerr << "MergeFarmCollector: Received nullptr task, indicating a "
                   "worker critical error or unexpected EOS path."
                << std::endl;
    }
    return GO_ON; // Collector is always ready for more until farm terminates
                  // it.
  }
  ~MergeFarmCollector() override = default;
};

// --- Main Parallel Merge Sort Function ---
void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads) {
  if (n_elements <= 1) {
    // Array is trivially sorted.
    return;
  }

  int actual_num_threads = num_threads;
  if (actual_num_threads <= 0) {
    actual_num_threads =
        ff_numCores(); // Use all available cores if not specified or invalid.
    if (actual_num_threads <= 0)
      actual_num_threads = 1; // Absolute fallback.
  }

  // For very small arrays or if only 1 thread is designated, use sequential
  // sort. The threshold considers that merge sort benefits from larger chunks.
  if (actual_num_threads == 1 || n_elements < INSERTION_SORT_THRESHOLD * 2) {
    sequential_merge_sort(records_array, n_elements, r_payload_size_bytes);
    return;
  }

  const size_t actual_record_size = sizeof(
      Record); // USE sizeof(Record) for consistency with Record* indexing

  // --- Phase 1: Parallel Sorting of Initial Chunks (using ff_farm) ---
  std::vector<SortChunkTask> initial_sort_tasks_storage;
  initial_sort_tasks_storage.reserve(actual_num_threads);
  // unique_ptr manages lifetime of temporary buffers for initial sort workers.
  std::vector<std::unique_ptr<Record[]>>
      worker_sort_temp_buffers_storage( // Changed char[] to Record[]
          actual_num_threads);
  const size_t max_initial_chunk_size =
      (n_elements + actual_num_threads - 1) / actual_num_threads;

  for (int i = 0; i < actual_num_threads; ++i) {
    try {
      // Allocate Record arrays for temp buffers
      worker_sort_temp_buffers_storage[i] =
          std::make_unique<Record[]>(max_initial_chunk_size);
    } catch (const std::bad_alloc &e) {
      std::cerr << "Error: Failed to allocate temp buffer (Record array) for "
                   "initial sort worker "
                << i << ". " << e.what() << std::endl;
      throw; // Propagate critical allocation failure.
    }
  }

  std::vector<MergeTask>
      current_level_chunks_storage; // Holds MergeTasks for the current merge
                                    // level.
  current_level_chunks_storage.reserve(actual_num_threads);
  // char *current_chunk_start_char_ptr = reinterpret_cast<char
  // *>(records_array); // REMOVED
  Record *current_chunk_start_ptr = records_array; // USE Record*
  size_t remaining_elements = n_elements;

  // Prepare tasks for initial sorting farm.
  for (int i = 0; i < actual_num_threads && remaining_elements > 0; ++i) {
    size_t elements_for_this_chunk = n_elements / actual_num_threads;
    if (static_cast<size_t>(i) < (n_elements % actual_num_threads))
      elements_for_this_chunk++;
    elements_for_this_chunk =
        std::min(remaining_elements, elements_for_this_chunk);
    if (elements_for_this_chunk == 0)
      continue;

    // Record *chunk_data_ptr = reinterpret_cast<Record
    // *>(current_chunk_start_char_ptr); // REMOVED
    Record *chunk_data_ptr = current_chunk_start_ptr; // USE Record*
    initial_sort_tasks_storage.emplace_back(
        chunk_data_ptr, elements_for_this_chunk, r_payload_size_bytes,
        worker_sort_temp_buffers_storage[i].get()); // Pass Record* temp buffer
    // These initial MergeTasks do not own data; they point into records_array.
    current_level_chunks_storage.emplace_back(
        chunk_data_ptr, elements_for_this_chunk, r_payload_size_bytes, false);

    // current_chunk_start_char_ptr += elements_for_this_chunk *
    // actual_record_size; // REMOVED
    current_chunk_start_ptr +=
        elements_for_this_chunk; // Advance Record* pointer
    remaining_elements -= elements_for_this_chunk;
  }

  if (initial_sort_tasks_storage
          .empty()) { // Should only occur if n_elements was 0.
    if (n_elements >
        0) { // Safety net if logic above had an issue for n_elements > 0
      sequential_merge_sort(records_array, n_elements, r_payload_size_bytes);
    }
    return;
  }

  // Execute the initial sorting farm.
  ff::ff_farm farm_sorter;
  TaskEmitter initial_emitter(initial_sort_tasks_storage);
  farm_sorter.add_emitter(
      &initial_emitter); // Emitter is stack/vector-based, not cleaned by farm.

  std::vector<ff::ff_node *> farm_sorter_workers_vec;
  farm_sorter_workers_vec.reserve(initial_sort_tasks_storage.size());
  for (size_t i = 0; i < initial_sort_tasks_storage.size(); ++i) {
    farm_sorter_workers_vec.push_back(new FFInitialSortWorker());
  }
  farm_sorter.add_workers(farm_sorter_workers_vec);
  farm_sorter.cleanup_workers(); // Farm will delete these dynamically allocated
                                 // workers.
  farm_sorter.remove_collector(); // Results are written in-place; no collection
                                  // needed from this farm.

  if (farm_sorter.run_and_wait_end() < 0) {
    std::cerr << "Error: FastFlow farm for initial sort failed." << std::endl;
    return; // Resources managed by farm_sorter's destructor via
            // cleanup_workers.
  }
  // At this point, records_array contains locally sorted chunks.
  // current_level_chunks_storage describes these chunks.

  // --- Phase 2: Iterative Parallel Merge of Sorted Chunks (using ff_farm for
  // each merge level) ---
  if (current_level_chunks_storage.size() <= 1) {
    // Array is already sorted (or was a single chunk).
    return;
  }

  // Each iteration of this loop performs one level of pairwise merges.
  while (current_level_chunks_storage.size() > 1) {
    std::vector<MergeTask>
        next_level_chunks_storage; // Stores results of this merge level.
    size_t num_pairs_to_merge = current_level_chunks_storage.size() / 2;

    if (num_pairs_to_merge == 0) {
      // This case implies current_level_chunks_storage.size() is 1 (odd one
      // from previous). The loop condition (size > 1) should prevent this if
      // logic is perfect, but as a safeguard, if only one chunk remains, move
      // it and break.
      if (current_level_chunks_storage.size() == 1) {
        next_level_chunks_storage.emplace_back(
            std::move(current_level_chunks_storage[0]));
      }
      current_level_chunks_storage = std::move(next_level_chunks_storage);
      break;
    }
    next_level_chunks_storage.reserve(
        num_pairs_to_merge + (current_level_chunks_storage.size() % 2));

    ff::ff_farm merge_farm_for_level(
        false); // false: farm does not create its own input channel.
    MergeFarmEmitter level_emitter_for_farm(
        current_level_chunks_storage); // Feeds pairs.
    MergeFarmCollector level_collector_for_farm(
        next_level_chunks_storage); // Collects merged results.

    merge_farm_for_level.add_emitter(&level_emitter_for_farm);
    merge_farm_for_level.add_collector(&level_collector_for_farm);

    int num_merge_farm_workers =
        std::min((size_t)actual_num_threads, num_pairs_to_merge);
    num_merge_farm_workers =
        std::max(1, num_merge_farm_workers); // Ensure at least one worker.

    std::vector<ff::ff_node *> merge_farm_workers_vec;
    merge_farm_workers_vec.reserve(num_merge_farm_workers);
    for (int i = 0; i < num_merge_farm_workers; ++i) {
      merge_farm_workers_vec.push_back(new MergeFarmWorker());
    }
    merge_farm_for_level.add_workers(merge_farm_workers_vec);
    merge_farm_for_level.cleanup_workers(); // Farm manages these workers.

    if (merge_farm_for_level.run_and_wait_end() < 0) {
      std::cerr << "Error: Merge farm for a level failed." << std::endl;
      // Destructors of MergeTasks in current_level_chunks_storage and
      // next_level_chunks_storage will attempt to clean up owned data.
      return;
    }

    // If there was an odd number of chunks, the emitter wouldn't have paired
    // the last one. Move this last remaining (unmerged) chunk to the next
    // level's input.
    if (level_emitter_for_farm.next_pair_start_idx <
        current_level_chunks_storage.size()) {
      // The remaining chunk is at
      // current_level_chunks_storage[level_emitter_for_farm.next_pair_start_idx]
      next_level_chunks_storage.emplace_back(
          std::move(current_level_chunks_storage[level_emitter_for_farm
                                                     .next_pair_start_idx]));
    }
    // current_level_chunks_storage has had its paired elements moved out.
    // It might be empty or contain one unmoved element if odd.
    // The 'next_level_chunks_storage' now holds all results for the next
    // iteration.
    current_level_chunks_storage = std::move(next_level_chunks_storage);
  }

  // --- Final step: Ensure result is in original records_array ---
  // After the loop, current_level_chunks_storage should contain exactly one
  // MergeTask.
  if (!current_level_chunks_storage.empty()) {
    if (current_level_chunks_storage[0].data_ptr != records_array) {
      // The final sorted data is in a dynamically allocated buffer owned by the
      // MergeTask. Copy it back to the original user-provided array.
      if (current_level_chunks_storage[0].data_ptr) {
        // std::memcpy(records_array, current_level_chunks_storage[0].data_ptr,
        //             n_elements * actual_record_size); // REPLACED with loop
        //             of copy_record
        for (size_t i = 0; i < n_elements; ++i) {
          copy_record(&records_array[i],
                      &current_level_chunks_storage[0].data_ptr[i],
                      r_payload_size_bytes);
        }
      } else if (n_elements > 0) {
        // This indicates an error if data_ptr is null but we expected elements.
        std::cerr << "Error: Final merge task has null data pointer but "
                     "non-zero global element count."
                  << std::endl;
      }
    }
    // The MergeTask at current_level_chunks_storage[0] (and its owned data if
    // any) will be cleaned up when current_level_chunks_storage goes out of
    // scope or is cleared.
  } else if (n_elements > 0) {
    // This signifies an issue if the original array was non-empty but the merge
    // process resulted in an empty list.
    std::cerr << "Error: Merge process resulted in an empty list of chunks for "
                 "a non-empty input."
              << std::endl;
  }
}
