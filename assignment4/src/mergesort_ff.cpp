// Project-specific headers - included first
#include "mergesort_ff.h" // Function signature for parallel_merge_sort_ff
#include "mergesort_common.h" // For sequential_merge_sort_recursive, Record, get_record_actual_size

// Standard C++ library headers
#include <algorithm> // For std::min, std::sort (potentially for small sorts or verification)
#include <iostream> // For std::cerr (error reporting)
#include <memory> // For std::unique_ptr (though raw pointers are used for ff_node for simplicity with ff_farm API)
#include <new> // For std::bad_alloc (exception for memory allocation failures)
#include <queue> // For std::priority_queue, essential for efficient K-way merge
#include <vector> // For std::vector, used for managing tasks and buffers

// FastFlow library main header
// This should pull in all necessary FastFlow components.
#include <ff/ff.hpp>
// Explicitly include components if ff.hpp is too minimal or for clarity.
// From previous logs, it seems direct includes of farm and node are safe.
#include <ff/farm.hpp>
#include <ff/node.hpp>

// --- Type and Structure Definitions ---

// SortChunkTask: Defines a unit of work for FastFlow workers.
// Each task represents a sub-array (chunk) to be sorted independently.
struct SortChunkTask {
  Record *data_ptr;                 // Pointer to the first Record in the chunk.
  size_t count;                     // Number of Records in this chunk.
  size_t r_payload_size_bytes_task; // Actual payload size for Records in this
                                    // chunk, crucial for memory operations.
  char *temp_buffer_for_worker;     // Pre-allocated temporary buffer for this
                                    // worker's sort, to avoid re-allocations.

  // Constructor initializes a sorting task.
  SortChunkTask(Record *ptr, size_t n, size_t r_size, char *temp_buf)
      : data_ptr(ptr), count(n), r_payload_size_bytes_task(r_size),
        temp_buffer_for_worker(temp_buf) {}
};

// --- FastFlow Node Implementations ---

// FFInitialSortWorker: A FastFlow worker node responsible for sorting an
// individual chunk of records. It derives from ff_node_t to process tasks of
// type SortChunkTask. The output type is also SortChunkTask, though in this
// farm configuration with a removed collector, the actual task object isn't
// passed on; instead, FF_GO_ON signals completion.
class FFInitialSortWorker : public ff::ff_node_t<SortChunkTask, SortChunkTask> {
public:
  FFInitialSortWorker() = default; // Default constructor.

  // svc: The core service method executed by the worker.
  // It receives a pointer to a SortChunkTask, sorts the corresponding data
  // chunk, and then signals its readiness for a new task.
  SortChunkTask *svc(SortChunkTask *task) override {
    // Precondition: task pointer and its data_ptr must be valid and count > 0.
    // If not, it indicates an issue upstream or an EOS marker that wasn't fully
    // handled.
    if (!task || !task->data_ptr || task->count == 0) {
      // Returning FF_GO_ON cast to the output type indicates that the worker
      // completed its operation (or had nothing to do) and is ready for more
      // tasks. This is crucial when the farm's collector is removed.
      return reinterpret_cast<SortChunkTask *>(ff::FF_GO_ON);
    }

    // The core work: sort the assigned chunk using the provided sequential
    // merge sort. The temporary buffer is passed to avoid repeated allocations
    // within the sort.
    sequential_merge_sort_recursive(task->data_ptr, task->count,
                                    task->r_payload_size_bytes_task,
                                    task->temp_buffer_for_worker);

    // Signal completion and readiness for a new task.
    return reinterpret_cast<SortChunkTask *>(ff::FF_GO_ON);
  }

  ~FFInitialSortWorker() override = default; // Default virtual destructor.
};

// TaskEmitter: A FastFlow emitter node.
// It's responsible for creating and sending SortChunkTask instances to the
// ff_farm workers. It effectively partitions the initial large array into
// smaller chunks.
class TaskEmitter : public ff::ff_node_t<SortChunkTask> { // Output-only node.
private:
  std::vector<SortChunkTask>
      &task_descriptors_ref; // Holds all tasks to be emitted. Passed by
                             // reference for efficiency.
  size_t next_task_idx;      // Tracks the next task to be dispatched.
  bool eos_sent; // Ensures End-Of-Stream (EOS) is emitted only once.

public:
  // Constructor: Initializes with a reference to the pre-computed task
  // descriptors.
  TaskEmitter(std::vector<SortChunkTask> &tasks)
      : task_descriptors_ref(tasks), next_task_idx(0), eos_sent(false) {}

  // svc: The core service method for the emitter.
  // It's called by the farm's load balancer to get the next task.
  // The input parameter (task_ignored) is not used by this emitter.
  SortChunkTask *svc(SortChunkTask * /*task_ignored*/) override {
    if (next_task_idx < task_descriptors_ref.size()) {
      // If there are more tasks, return a pointer to the next one.
      return &task_descriptors_ref[next_task_idx++];
    }

    if (!eos_sent) {
      // Once all tasks are dispatched, send an EOS signal to the farm.
      eos_sent = true;
      // EOS is a special void* token, cast to the node's output type.
      return reinterpret_cast<SortChunkTask *>(EOS);
    }

    // After EOS, if svc is called again, return nullptr to signify no more
    // tasks.
    return nullptr;
  }
  ~TaskEmitter() override = default; // Default virtual destructor.
};

// --- Main Parallel Merge Sort Function ---

// parallel_merge_sort_ff: Implements the parallel merge sort using FastFlow.
// 1. Divides the input array into 'num_threads' chunks.
// 2. Uses an ff_farm to sort these chunks in parallel.
// 3. The main thread then performs a K-way merge on the sorted chunks.
void parallel_merge_sort_ff(Record *records_array, size_t n_elements,
                            size_t r_payload_size_bytes, int num_threads) {
  // Handle trivial cases: empty or single-element array.
  if (n_elements <= 1) {
    return; // Already sorted.
  }

  // If num_threads is 1 or less, parallel overhead is not justified.
  // Fallback to a purely sequential version for efficiency.
  if (num_threads <= 1) {
    sequential_merge_sort(records_array, n_elements, r_payload_size_bytes);
    return;
  }

  const size_t actual_record_size =
      get_record_actual_size(r_payload_size_bytes);

  // --- Phase 1: Parallel Sorting of Chunks ---

  // Store descriptors for each chunk/task. These are fed by the Emitter.
  // The vector itself is on the stack, ensuring its lifetime covers the farm's
  // execution.
  std::vector<SortChunkTask> task_descriptors_storage;
  task_descriptors_storage.reserve(
      num_threads); // Pre-allocate to avoid vector reallocations.

  // Allocate temporary buffers for each worker.
  // Each worker gets its own buffer to prevent contention and simplify
  // management. The size is based on the largest possible chunk a worker might
  // receive.
  std::vector<std::vector<char>> worker_temp_buffers(num_threads);
  const size_t max_chunk_size_elements =
      (n_elements + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    try {
      worker_temp_buffers[i].resize(max_chunk_size_elements *
                                    actual_record_size);
    } catch (const std::bad_alloc &e) {
      // Handle memory allocation failure, which is critical.
      std::cerr << "Error: Failed to allocate temporary buffer for worker " << i
                << ". Requested size: "
                << (max_chunk_size_elements * actual_record_size)
                << " bytes. Exception: " << e.what() << std::endl;
      throw; // Re-throw to signal critical failure.
    }
  }

  // Partition the main array into chunks and create task descriptors.
  char *current_chunk_start_ptr_char = reinterpret_cast<char *>(records_array);
  size_t remaining_elements = n_elements;

  for (int i = 0; i < num_threads && remaining_elements > 0; ++i) {
    size_t elements_for_this_chunk = n_elements / num_threads;
    if (static_cast<size_t>(i) < (n_elements % num_threads)) {
      elements_for_this_chunk++;
    }
    elements_for_this_chunk =
        std::min(remaining_elements, elements_for_this_chunk);

    if (elements_for_this_chunk == 0)
      continue; // Skip if no elements for this chunk.

    task_descriptors_storage.emplace_back(
        reinterpret_cast<Record *>(current_chunk_start_ptr_char),
        elements_for_this_chunk, r_payload_size_bytes,
        worker_temp_buffers[i]
            .data() // Pass the dedicated temp buffer to the task.
    );
    current_chunk_start_ptr_char +=
        elements_for_this_chunk * actual_record_size;
    remaining_elements -= elements_for_this_chunk;
  }

  // Setup and run the FastFlow farm.
  ff::ff_farm farm; // Create the farm object.
  TaskEmitter emitter(
      task_descriptors_storage); // Emitter uses the pre-calculated tasks.
  farm.add_emitter(
      &emitter); // Emitter object is on the stack, passed by pointer.

  std::vector<ff::ff_node *> workers_ptr_vec; // Vector to hold worker pointers.
  workers_ptr_vec.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // Workers are dynamically allocated.
    workers_ptr_vec.push_back(new FFInitialSortWorker());
  }
  farm.add_workers(workers_ptr_vec); // Add workers to the farm.
  // Instruct the farm to manage the lifecycle of these dynamically allocated
  // workers. This means the farm's destructor will delete them.
  farm.cleanup_workers();

  // The collector is explicitly removed because the final merge (K-way merge)
  // will be performed by the main thread after the farm completes its
  // chunk-sorting phase. This simplifies the farm logic as workers don't need
  // to output their sorted chunks to a collector node within the FastFlow
  // graph.
  farm.remove_collector();

  // Execute the farm and wait for all tasks (chunk sorts) to complete.
  if (farm.run_and_wait_end() < 0) {
    std::cerr
        << "Error: FastFlow farm execution failed during chunk sorting phase."
        << std::endl;
    // farm.cleanup_workers() ensures workers are deleted even on error if farm
    // was constructed.
    return;
  }

  // --- Phase 2: K-Way Merge of Sorted Chunks by the Main Thread ---
  // This phase is necessary only if there was more than one chunk to sort.
  if (task_descriptors_storage.size() > 1) {
    char *final_merge_temp_buffer_raw = nullptr;
    try {
      // Allocate a single large temporary buffer for the final K-way merge.
      // This buffer will hold the entire sorted array temporarily.
      final_merge_temp_buffer_raw = new char[n_elements * actual_record_size];
    } catch (const std::bad_alloc &e) {
      std::cerr << "Error: Failed to allocate final merge temporary buffer. "
                   "Requested size: "
                << (n_elements * actual_record_size)
                << " bytes. Exception: " << e.what() << std::endl;
      throw;
    }

    // PQueueNode: Helper struct for the K-way merge priority queue.
    // Stores a pointer to the current record from a chunk and metadata.
    struct PQueueNode {
      const Record *current_record_ptr; // The actual record data (current
                                        // smallest in its chunk).
      size_t chunk_source_idx; // Identifier for the source chunk (mainly for
                               // debugging/tracking).
      const Record *next_element_in_chunk_ptr; // Pointer to the next Record in
                                               // this chunk.
      const Record *chunk_end_ptr; // Pointer marking the end of this chunk (one
                                   // past the last element).
      size_t r_payload_size_node;  // Payload size for records from this chunk.

      // Custom comparator for the min-priority queue, orders by Record key.
      bool operator>(const PQueueNode &other) const {
        return current_record_ptr->key > other.current_record_ptr->key;
      }
    };

    // Min-priority queue to manage the current smallest element from each of
    // the K sorted chunks.
    std::priority_queue<PQueueNode, std::vector<PQueueNode>,
                        std::greater<PQueueNode>>
        pq;

    // Initialize the priority queue with the first element from each sorted
    // chunk.
    for (size_t i = 0; i < task_descriptors_storage.size(); ++i) {
      if (task_descriptors_storage[i].count >
          0) { // Only consider non-empty chunks.
        const Record *chunk_data_start_ptr =
            task_descriptors_storage[i].data_ptr;
        const size_t chunk_record_mem_size = get_record_actual_size(
            task_descriptors_storage[i].r_payload_size_bytes_task);

        const Record *chunk_logical_end_ptr = reinterpret_cast<const Record *>(
            reinterpret_cast<const char *>(chunk_data_start_ptr) +
            task_descriptors_storage[i].count * chunk_record_mem_size);

        // Pointer to the element *after* the current_record_ptr in this chunk.
        const Record *first_element_next_ptr = reinterpret_cast<const Record *>(
            reinterpret_cast<const char *>(chunk_data_start_ptr) +
            chunk_record_mem_size);

        pq.push({chunk_data_start_ptr,   // current_record_ptr
                 i,                      // chunk_source_idx
                 first_element_next_ptr, // next_element_in_chunk_ptr
                 chunk_logical_end_ptr,  // chunk_end_ptr
                 task_descriptors_storage[i]
                     .r_payload_size_bytes_task}); // r_payload_size_node
      }
    }

    char *merged_output_write_ptr =
        final_merge_temp_buffer_raw; // Current write position in the final
                                     // merge buffer.
    size_t merged_records_count = 0;

    // Repeatedly extract the minimum element from the priority queue and add it
    // to the result.
    while (!pq.empty() && merged_records_count < n_elements) {
      PQueueNode smallest_node =
          pq.top(); // Get the chunk with the current overall smallest element.
      pq.pop();

      const size_t record_mem_size_for_current_node =
          get_record_actual_size(smallest_node.r_payload_size_node);

      // Copy the smallest record to the final merge buffer.
      std::memcpy(merged_output_write_ptr, smallest_node.current_record_ptr,
                  record_mem_size_for_current_node);
      merged_output_write_ptr += record_mem_size_for_current_node;
      merged_records_count++;

      // If the chunk from which the element was taken still has more elements,
      // add the next element from that chunk back into the priority queue.
      if (smallest_node.next_element_in_chunk_ptr <
          smallest_node.chunk_end_ptr) {
        const Record *new_node_current_ptr =
            smallest_node.next_element_in_chunk_ptr;
        const Record *new_node_next_ptr = reinterpret_cast<const Record *>(
            reinterpret_cast<const char *>(new_node_current_ptr) +
            record_mem_size_for_current_node);
        pq.push({new_node_current_ptr, smallest_node.chunk_source_idx,
                 new_node_next_ptr, smallest_node.chunk_end_ptr,
                 smallest_node.r_payload_size_node});
      }
    }

    // Copy the fully sorted data from the temporary merge buffer back to the
    // original array.
    std::memcpy(records_array, final_merge_temp_buffer_raw,
                n_elements * actual_record_size);
    delete[] final_merge_temp_buffer_raw; // Deallocate the K-way merge
                                          // temporary buffer.
  }
  // If task_descriptors_storage.size() <= 1, the array was sorted by a single
  // worker (or sequentially if num_threads <=1), so it's already in its final
  // sorted state in records_array.
}
