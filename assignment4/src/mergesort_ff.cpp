#include "mergesort_ff.h"
#include "mergesort_common.h"
#include "record.h"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
#include <vector>

using namespace ff;

// Task structures
struct SortJob {
  Record *data;
  size_t start;
  size_t size;
  size_t payload_size;

  SortJob(Record *data, size_t start, size_t size, size_t payload_size)
      : data(data), start(start), size(size), payload_size(payload_size) {}
};

struct SortedRun {
  Record *data;
  size_t start;
  size_t size;
  size_t payload_size;

  SortedRun(Record *data, size_t start, size_t size, size_t payload_size)
      : data(data), start(start), size(size), payload_size(payload_size) {}
};

// Worker node for initial sorting
class SorterNode : public ff_node_t<SortJob> {
public:
  SortJob *svc(SortJob *job) {
    if (job == nullptr || job == EOS || job == GO_ON)
      return job;

    // Sort the chunk in-place
    sequential_sort_records(job->data + job->start, job->size,
                            job->payload_size);

    return job; // Return the same job, now with sorted data
  }
};

void parallel_merge_sort_ff(Record data[], size_t n, size_t payload_size,
                            int num_threads) {
  if (n <= 1)
    return;

  // Calculate chunk size for initial sorting
  size_t chunk_size = std::max((size_t)1, n / (size_t)num_threads);

  // Create the sorting farm
  ff_farm sort_farm;
  std::vector<ff_node *> sort_workers;
  for (int i = 0; i < num_threads; i++) {
    sort_workers.push_back(new SorterNode());
  }
  sort_farm.add_workers(sort_workers);
  sort_farm.cleanup_workers(); // FastFlow will delete workers

  // Run the farm in accelerator mode
  sort_farm.run_and_wait_end();

  // Submit initial sort jobs
  std::vector<SortedRun> sorted_runs;
  for (size_t i = 0; i < n; i += chunk_size) {
    size_t current_chunk_size = std::min(chunk_size, n - i);
    auto *job = new SortJob(data, i, current_chunk_size, payload_size);
    sort_farm.offload(job);
  }
  sort_farm.offload(sort_farm.EOS);

  // Collect sorted chunks
  void *result = nullptr;
  while (sort_farm.load_result(&result)) {
    if (result != sort_farm.EOS) {
      SortJob *job = static_cast<SortJob *>(result);
      sorted_runs.emplace_back(job->data, job->start, job->size,
                               job->payload_size);
      delete job;
    }
  }

  sort_farm.wait();

  // Phase 2: Merge sorted runs iteratively
  // Create temporary buffer for merging
  Record *temp_buffer = new Record[n];
  // No need to allocate rpayload since it's a fixed-size array

  // Iteratively merge runs until only one remains
  while (sorted_runs.size() > 1) {
    std::vector<SortedRun> next_level;

    for (size_t i = 0; i < sorted_runs.size(); i += 2) {
      if (i + 1 < sorted_runs.size()) {
        // Merge two runs
        const auto &run1 = sorted_runs[i];
        const auto &run2 = sorted_runs[i + 1];

        merge_two_sorted_runs(run1.data + run1.start, run1.size,
                              run2.data + run2.start, run2.size, temp_buffer,
                              payload_size);

        // Copy merged result back to original location
        size_t merged_size = run1.size + run2.size;
        for (size_t j = 0; j < merged_size; j++) {
          copy_record_payload_aware(run1.data + run1.start + j, temp_buffer + j,
                                    payload_size);
        }

        next_level.emplace_back(run1.data, run1.start, merged_size,
                                payload_size);
      } else {
        // Odd run, just add it to next level
        next_level.push_back(sorted_runs[i]);
      }
    }

    sorted_runs = std::move(next_level);
  }

  // Cleanup temporary buffer
  delete[] temp_buffer;
}
