#include "mergesort_ff.h"
#include "mergesort_common.h"

#include <algorithm>
#include <iostream>
#include <memory> // For std::unique_ptr, std::make_unique (though workers will be raw ptrs)
#include <new>
#include <queue>
#include <vector>

// FastFlow headers: ff.hpp first, then specific patterns.
#include <ff/farm.hpp> // For ff_farm
#include <ff/ff.hpp>   // Core FastFlow functionalities
#include <ff/node.hpp> // Base for ff_node_t

// It seems graph_utils.hpp and pipeline.hpp might be needed internally by
// FastFlow's own headers. Including them explicitly here is a precautionary
// measure if FastFlow's internal includes are not perfectly ordered for all
// compilers/versions.
#include <ff/graph_utils.hpp>
#include <ff/pipeline.hpp>

// Helper structure to define a sorting task for a FastFlow worker.
struct SortChunkTask {
  Record *data_ptr;
  size_t count;
  size_t r_payload_size_bytes_task;
  char *temp_buffer_for_worker;

  SortChunkTask(Record *ptr, size_t n, size_t r_size, char *temp_buf)
      : data_ptr(ptr), count(n), r_payload_size_bytes_task(r_size),
        temp_buffer_for_worker(temp_buf) {}
};

// FastFlow worker node for sorting individual chunks of records.
// Inherits from ff_node_t for typed input/output.
class FFInitialSortWorker : public ff::ff_node_t<SortChunkTask, SortChunkTask> {
public:
  FFInitialSortWorker() = default;

  SortChunkTask *svc(SortChunkTask *task) override {
    if (!task || !task->data_ptr || task->count == 0) {
      return task;
    }
    sequential_merge_sort_recursive(task->data_ptr, task->count,
                                    task->r_payload_size_bytes_task,
                                    task->temp_buffer_for_worker);
    return task;
  }
  virtual ~FFInitialSortWorker() = default;
};

// Custom Emitter node for FastFlow.
// This node is responsible for feeding the pre-defined SortChunkTask objects
// to the workers in the farm. It outputs tasks of type SortChunkTask.
class TaskEmitter : public ff::ff_node_t<SortChunkTask> { // Output-only node
                                                          // for SortChunkTask
private:
  std::vector<SortChunkTask>
      &task_descriptors_ref; // Reference to the vector of tasks.
  size_t next_task_idx;      // Tracks the next task to send.
  bool eos_sent;             // Ensures EOS is sent only once.

public:
  TaskEmitter(std::vector<SortChunkTask> &tasks)
      : task_descriptors_ref(tasks), next_task_idx(0), eos_sent(false) {}

  // The svc method for an emitter generates or retrieves tasks.
  // The input to svc (SortChunkTask*) is ignored as this emitter produces its
  // own stream.
  SortChunkTask *svc(SortChunkTask * /*unused_input_task*/) override {
    if (next_task_idx < task_descriptors_ref.size()) {
      // Return a pointer to the next task descriptor from the storage.
      return &task_descriptors_ref[next_task_idx++];
    }
    if (!eos_sent) {
      eos_sent = true;
      return EOS; // Signal End-Of-Stream once all tasks are emitted.
    }
    return nullptr; // After EOS, return nullptr if svc is called again.
  }
};

void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads) {
  if (n_elements <= 1) {
    return;
  }
  if (num_threads <= 1) {
    sequential_merge_sort(records_array, n_elements, r_payload_size_bytes);
    return;
  }

  size_t actual_record_size = get_record_actual_size(r_payload_size_bytes);

  // Storage for task descriptors. Their lifetime must cover the farm's
  // execution.
  std::vector<SortChunkTask> task_descriptors_storage;
  task_descriptors_storage.reserve(num_threads);

  // Pre-allocate temporary buffers for each worker.
  std::vector<std::vector<char>> worker_temp_buffers(num_threads);
  size_t max_chunk_size_elements = (n_elements + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    try {
      worker_temp_buffers[i].resize(max_chunk_size_elements *
                                    actual_record_size);
    } catch (const std::bad_alloc &e) {
      std::cerr << "Error: Failed to allocate temp buffer for worker " << i
                << ". " << e.what() << std::endl;
      throw;
    }
  }

  char *current_chunk_start_ptr = reinterpret_cast<char *>(records_array);
  size_t remaining_elements = n_elements;

  for (int i = 0; i < num_threads && remaining_elements > 0; ++i) {
    size_t elements_for_this_chunk =
        (n_elements / num_threads) +
        (static_cast<size_t>(i) < (n_elements % num_threads) ? 1 : 0);
    elements_for_this_chunk =
        std::min(remaining_elements, elements_for_this_chunk);

    if (elements_for_this_chunk == 0)
      continue;

    task_descriptors_storage.emplace_back(
        reinterpret_cast<Record *>(current_chunk_start_ptr),
        elements_for_this_chunk, r_payload_size_bytes,
        worker_temp_buffers[i].data());
    current_chunk_start_ptr += elements_for_this_chunk * actual_record_size;
    remaining_elements -= elements_for_this_chunk;
  }

  // Create FastFlow farm and components, similar to the mandelbrot example.
  ff::ff_farm farm; // Use the non-templatized ff_farm.

  TaskEmitter emitter(task_descriptors_storage);
  farm.add_emitter(&emitter); // Add custom emitter.

  std::vector<ff::ff_node *> workers_ptr_vec;
  workers_ptr_vec.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // Workers are created with 'new'. The ff_farm is expected to manage their
    // deletion if its cleanup_workers flag is true (default for
    // add_workers(vector<ff_node*>&)).
    workers_ptr_vec.push_back(new FFInitialSortWorker());
  }
  farm.add_workers(workers_ptr_vec); // Add vector of raw worker pointers.
  farm.remove_collector(); // K-way merge will be done by the main thread.

  if (farm.run_and_wait_end() < 0) {
    std::cerr << "Error: FastFlow farm execution failed." << std::endl;
    // If run_and_wait_end fails early, manual cleanup of workers might be
    // needed if farm didn't take full ownership or had an issue. However,
    // typically the farm's destructor handles workers added via add_workers. To
    // be extremely safe if cleanup is a concern upon error: for(ff::ff_node*
    // w_ptr : workers_ptr_vec) delete w_ptr;
    return;
  }

  // --- Phase 2: K-Way Merge of Sorted Chunks ---
  if (task_descriptors_storage.size() > 1) {
    char *final_merge_temp_buffer_raw = nullptr;
    try {
      final_merge_temp_buffer_raw = new char[n_elements * actual_record_size];
    } catch (const std::bad_alloc &e) {
      std::cerr << "Error: Failed to allocate final merge temp buffer. "
                << e.what() << std::endl;
      throw;
    }

    struct PQueueNode {
      const Record *current_record_ptr;
      size_t chunk_idx;
      const Record *chunk_current_end_ptr;
      size_t r_payload_size_node;

      bool operator>(const PQueueNode &other) const {
        return current_record_ptr->key > other.current_record_ptr->key;
      }
    };

    std::priority_queue<PQueueNode, std::vector<PQueueNode>,
                        std::greater<PQueueNode>>
        pq;
    size_t current_chunk_record_mem_size;

    for (size_t i = 0; i < task_descriptors_storage.size(); ++i) {
      if (task_descriptors_storage[i].count > 0) {
        current_chunk_record_mem_size = get_record_actual_size(
            task_descriptors_storage[i].r_payload_size_bytes_task);
        const Record *chunk_start = task_descriptors_storage[i].data_ptr;
        const Record *chunk_end = reinterpret_cast<const Record *>(
            reinterpret_cast<const char *>(chunk_start) +
            task_descriptors_storage[i].count * current_chunk_record_mem_size);
        pq.push({chunk_start, i, chunk_end,
                 task_descriptors_storage[i].r_payload_size_bytes_task});
      }
    }

    char *merged_output_ptr_char = final_merge_temp_buffer_raw;
    size_t merged_count = 0;

    while (!pq.empty() && merged_count < n_elements) {
      PQueueNode smallest_node = pq.top();
      pq.pop();

      current_chunk_record_mem_size =
          get_record_actual_size(smallest_node.r_payload_size_node);
      std::memcpy(merged_output_ptr_char, smallest_node.current_record_ptr,
                  current_chunk_record_mem_size);
      merged_output_ptr_char += current_chunk_record_mem_size;
      merged_count++;

      const Record *next_in_chunk = reinterpret_cast<const Record *>(
          reinterpret_cast<const char *>(smallest_node.current_record_ptr) +
          current_chunk_record_mem_size);

      if (next_in_chunk < smallest_node.chunk_current_end_ptr) {
        pq.push({next_in_chunk, smallest_node.chunk_idx,
                 smallest_node.chunk_current_end_ptr,
                 smallest_node.r_payload_size_node});
      }
    }

    std::memcpy(records_array, final_merge_temp_buffer_raw,
                n_elements * actual_record_size);
    delete[] final_merge_temp_buffer_raw;
  }
}
