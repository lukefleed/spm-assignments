#include "../common/timer.hpp" // Explicitly include timer
#include "../common/utils.hpp" // Assumed to contain Record, Config, Timer, etc.
#include <algorithm>
#include <ff/ff.hpp>
#include <iostream> // For debugging prints
#include <memory>
#include <queue>
#include <vector>

using namespace ff;

// ====== STRUCTURES FOR PIPELINE ======
struct SortChunk {
  std::vector<Record> data;
  size_t chunk_id;
  size_t level;

  SortChunk(std::vector<Record> d, size_t id, size_t lvl = 0)
      : data(std::move(d)), chunk_id(id), level(lvl) {}
};

struct MergePair {
  std::vector<Record> left;
  std::vector<Record> right;
  size_t result_id;
  size_t level;

  MergePair(std::vector<Record> l, std::vector<Record> r, size_t id, size_t lvl)
      : left(std::move(l)), right(std::move(r)), result_id(id), level(lvl) {}
};

// ====== FIRST FARM: SORTING ======
class SortEmitter : public ff_node {
private:
  std::vector<Record> &original_data_ref;
  size_t num_workers_in_farm;
  size_t chunk_size_val;
  size_t current_chunk_idx;
  size_t total_chunks_to_emit_val;
  std::vector<SortChunk> prepared_chunks_val;

public:
  SortEmitter(std::vector<Record> &data, size_t workers_count)
      : original_data_ref(data), num_workers_in_farm(workers_count),
        current_chunk_idx(0) {
    if (original_data_ref.empty()) {
      chunk_size_val = 0;
      total_chunks_to_emit_val = 0;
    } else {
      size_t effective_workers = std::max((size_t)1, num_workers_in_farm);
      chunk_size_val = (original_data_ref.size() + effective_workers - 1) /
                       effective_workers;
      chunk_size_val = std::max((size_t)1, chunk_size_val);

      total_chunks_to_emit_val =
          (original_data_ref.size() + chunk_size_val - 1) / chunk_size_val;
    }
    std::cerr << "[SortEmitter] Constructor: original_data_ref.size() = "
              << original_data_ref.size()
              << ", workers_count = " << workers_count
              << ", chunk_size_val = " << chunk_size_val
              << ", total_chunks_to_emit_val = " << total_chunks_to_emit_val
              << std::endl;

    prepared_chunks_val.reserve(total_chunks_to_emit_val);
    for (size_t i = 0; i < total_chunks_to_emit_val; ++i) {
      size_t start = i * chunk_size_val;
      size_t end = std::min(start + chunk_size_val, original_data_ref.size());
      std::vector<Record> chunk_data_segment;
      if (start < end) {
        chunk_data_segment.reserve(end - start);
        for (size_t j = start; j < end; ++j) {
          chunk_data_segment.emplace_back(std::move(original_data_ref[j]));
        }
      }
      prepared_chunks_val.emplace_back(std::move(chunk_data_segment), i);
    }
    if (total_chunks_to_emit_val > 0) {
      std::cerr << "[SortEmitter] Constructor: Clearing original_data_ref "
                   "after moving records. Original size before clear: "
                << original_data_ref.size() << std::endl;
      original_data_ref.clear();
      std::cerr
          << "[SortEmitter] Constructor: original_data_ref size after clear: "
          << original_data_ref.size() << std::endl;
    } else {
      std::cerr << "[SortEmitter] Constructor: No chunks to emit, "
                   "original_data_ref not cleared by emitter. Size: "
                << original_data_ref.size() << std::endl;
    }
  }

  void *svc(void * /*task*/) override {
    if (current_chunk_idx >= total_chunks_to_emit_val) {
      std::cerr << "[SortEmitter] SVC: Emitted all " << total_chunks_to_emit_val
                << " chunks. Sending EOS." << std::endl;
      return EOS;
    }
    std::cerr << "[SortEmitter] SVC: Emitting chunk "
              << prepared_chunks_val[current_chunk_idx].chunk_id << " (idx "
              << current_chunk_idx << ")" << std::endl;
    auto *chunk_task =
        new SortChunk(std::move(prepared_chunks_val[current_chunk_idx].data),
                      prepared_chunks_val[current_chunk_idx].chunk_id,
                      prepared_chunks_val[current_chunk_idx].level);
    current_chunk_idx++;
    return chunk_task;
  }

  size_t get_total_chunks() const { return total_chunks_to_emit_val; }
};

class SortWorker : public ff_node_t<SortChunk, SortChunk> {
public:
  SortChunk *svc(SortChunk *chunk) override {
    if (chunk == (SortChunk *)EOS) {
      return (SortChunk *)EOS;
    }
    std::cerr << "[SortWorker " << get_my_id() << "] SVC: Received chunk "
              << chunk->chunk_id << " with " << chunk->data.size()
              << " records. Sorting." << std::endl;
    std::sort(chunk->data.begin(), chunk->data.end());
    std::cerr << "[SortWorker " << get_my_id()
              << "] SVC: Finished sorting chunk " << chunk->chunk_id << "."
              << std::endl;
    return chunk;
  }
};

class SortCollector : public ff_node_t<SortChunk, SortChunk> {
private:
  std::vector<SortChunk *> collected_chunks_buffer;
  size_t num_expected_chunks;
  size_t num_received_data_chunks;
  size_t num_sent_chunks;
  bool upstream_farm_eos_received;

public:
  SortCollector(size_t expected_total_chunks)
      : num_expected_chunks(expected_total_chunks), num_received_data_chunks(0),
        num_sent_chunks(0), upstream_farm_eos_received(false) {
    std::cerr << "[SortCollector] Constructor: expecting "
              << num_expected_chunks << " chunks." << std::endl;
  }

  SortChunk *svc(SortChunk *chunk) override {
    bool is_data_signal = (chunk != (SortChunk *)EOS &&
                           chunk != (SortChunk *)GO_ON && chunk != nullptr);

    if (is_data_signal) {
      std::cerr << "[SortCollector] SVC: DATA chunk " << chunk->chunk_id
                << " received (total data chunks received: "
                << num_received_data_chunks + 1 << ")" << std::endl;
      collected_chunks_buffer.push_back(chunk);
      num_received_data_chunks++;
    } else if (chunk == (SortChunk *)EOS) {
      std::cerr << "[SortCollector] SVC: EOS from upstream farm received."
                << std::endl;
      upstream_farm_eos_received = true;
    }

    if (num_sent_chunks < collected_chunks_buffer.size()) {
      std::cerr << "[SortCollector] SVC: Sending buffered chunk "
                << collected_chunks_buffer[num_sent_chunks]->chunk_id
                << " (sent " << num_sent_chunks + 1 << " of "
                << collected_chunks_buffer.size() << " buffered)" << std::endl;
      return collected_chunks_buffer[num_sent_chunks++];
    }

    if (upstream_farm_eos_received &&
        num_sent_chunks == collected_chunks_buffer.size() &&
        num_received_data_chunks >= num_expected_chunks) {
      std::cerr << "[SortCollector] SVC: Propagating EOS. (expected="
                << num_expected_chunks
                << ", received_data=" << num_received_data_chunks
                << ", sent=" << num_sent_chunks << ")" << std::endl;
      return (SortChunk *)EOS;
    }

    std::cerr << "[SortCollector] SVC: Sending GO_ON. (expected="
              << num_expected_chunks
              << ", received_data=" << num_received_data_chunks
              << ", sent=" << num_sent_chunks
              << ", upstream_eos=" << upstream_farm_eos_received << ")"
              << std::endl;
    return (SortChunk *)GO_ON;
  }
};

// ====== SECOND FARM: MERGING ======
class MergeEmitter : public ff_node_t<SortChunk, MergePair> {
private:
  std::vector<SortChunk *> pending_chunks_to_pair;
  size_t next_pair_id;
  bool upstream_eos_received;

public:
  MergeEmitter() : next_pair_id(0), upstream_eos_received(false) {
    std::cerr << "[MergeEmitter] Constructor." << std::endl;
  }

  MergePair *svc(SortChunk *chunk) override {
    if (chunk != (SortChunk *)EOS && chunk != (SortChunk *)GO_ON &&
        chunk != nullptr) {
      std::cerr << "[MergeEmitter] SVC: DATA chunk " << chunk->chunk_id
                << " from SortCollector. Pending list size before add: "
                << pending_chunks_to_pair.size() << "." << std::endl;
      pending_chunks_to_pair.push_back(chunk);
    } else if (chunk == (SortChunk *)EOS) {
      std::cerr << "[MergeEmitter] SVC: EOS from SortCollector received."
                << std::endl;
      upstream_eos_received = true;
    }

    return try_create_or_forward_pair();
  }

private:
  MergePair *try_create_or_forward_pair() {
    if (pending_chunks_to_pair.size() >= 2) {
      std::cerr << "[MergeEmitter] try_create_or_forward_pair: Have "
                << pending_chunks_to_pair.size()
                << " pending chunks. Creating pair." << std::endl;
      SortChunk *left_chunk = pending_chunks_to_pair[0];
      SortChunk *right_chunk = pending_chunks_to_pair[1];

      pending_chunks_to_pair.erase(pending_chunks_to_pair.begin(),
                                   pending_chunks_to_pair.begin() + 2);

      auto *pair_task = new MergePair(
          std::move(left_chunk->data), std::move(right_chunk->data),
          next_pair_id++, std::max(left_chunk->level, right_chunk->level) + 1);
      std::cerr
          << "[MergeEmitter] try_create_or_forward_pair: Created MergePair "
          << pair_task->result_id << " from chunks " << left_chunk->chunk_id
          << " and " << right_chunk->chunk_id << std::endl;
      delete left_chunk;
      delete right_chunk;
      return pair_task;
    }

    if (upstream_eos_received) {
      if (pending_chunks_to_pair.size() == 1) {
        std::cerr << "[MergeEmitter] try_create_or_forward_pair: Upstream EOS "
                     "received. One chunk ("
                  << pending_chunks_to_pair[0]->chunk_id
                  << ") remaining. Forwarding." << std::endl;
        SortChunk *final_chunk = pending_chunks_to_pair[0];
        pending_chunks_to_pair.clear();

        auto *pair_task =
            new MergePair(std::move(final_chunk->data), std::vector<Record>(),
                          next_pair_id++, final_chunk->level);
        delete final_chunk;
        return pair_task;
      }
      if (pending_chunks_to_pair.empty()) {
        std::cerr << "[MergeEmitter] try_create_or_forward_pair: Upstream EOS "
                     "received. No chunks remaining. Sending EOS."
                  << std::endl;
        return (MergePair *)EOS;
      }
    }
    std::cerr
        << "[MergeEmitter] try_create_or_forward_pair: Sending GO_ON. Pending: "
        << pending_chunks_to_pair.size()
        << ", Upstream EOS: " << upstream_eos_received << std::endl;
    return (MergePair *)GO_ON;
  }
};

class MergeWorker : public ff_node_t<MergePair, SortChunk> {
public:
  SortChunk *svc(MergePair *pair) override {
    if (pair == (MergePair *)EOS) {
      return (SortChunk *)EOS;
    }
    std::cerr << "[MergeWorker " << get_my_id() << "] SVC: Received MergePair "
              << pair->result_id << ". Merging." << std::endl;

    std::vector<Record> merged_data;
    if (pair->right.empty()) {
      merged_data = std::move(pair->left);
    } else {
      merge_two_vectors(pair->left, pair->right, merged_data);
    }

    auto *result_chunk =
        new SortChunk(std::move(merged_data), pair->result_id, pair->level);
    std::cerr << "[MergeWorker " << get_my_id()
              << "] SVC: Finished merging pair " << pair->result_id
              << ". Result chunk has " << result_chunk->data.size()
              << " records." << std::endl;
    delete pair;
    return result_chunk;
  }

private:
  void merge_two_vectors(std::vector<Record> &left, std::vector<Record> &right,
                         std::vector<Record> &result) {
    result.reserve(left.size() + right.size());
    size_t i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
      if (left[i] <= right[j]) {
        result.push_back(std::move(left[i++]));
      } else {
        result.push_back(std::move(right[j++]));
      }
    }
    while (i < left.size()) {
      result.push_back(std::move(left[i++]));
    }
    while (j < right.size()) {
      result.push_back(std::move(right[j++]));
    }
  }
};

class MergeCollector : public ff_node_t<SortChunk, void> {
private:
  std::vector<Record> *final_result_ptr;
  std::vector<SortChunk *> collected_merged_chunks;
  bool merge_farm_eos_received;

public:
  MergeCollector(std::vector<Record> *result)
      : final_result_ptr(result), merge_farm_eos_received(false) {
    std::cerr << "[MergeCollector] Constructor." << std::endl;
  }

  void *svc(SortChunk *chunk) override {
    if (chunk != (SortChunk *)EOS && chunk != (SortChunk *)GO_ON &&
        chunk != nullptr) {
      std::cerr << "[MergeCollector] SVC: DATA chunk " << chunk->chunk_id
                << " (level " << chunk->level << ") with " << chunk->data.size()
                << " records. Accumulating." << std::endl;
      collected_merged_chunks.push_back(chunk);
      return GO_ON;
    } else if (chunk == (SortChunk *)EOS) {
      std::cerr << "[MergeCollector] SVC: EOS from merge farm received."
                << std::endl;
      merge_farm_eos_received = true;
    }

    if (merge_farm_eos_received) {
      std::cerr << "[MergeCollector] SVC: Merge farm EOS confirmed. Performing "
                   "final k-way merge with "
                << collected_merged_chunks.size() << " collected chunks."
                << std::endl;
      perform_k_way_merge();
      std::cerr
          << "[MergeCollector] SVC: Final merge complete. final_result_ptr has "
          << (final_result_ptr ? final_result_ptr->size() : 0) << " records."
          << std::endl;
      if (final_result_ptr && final_result_ptr->size() > 0 &&
          final_result_ptr->size() < 20) {
        std::cerr << "Final sorted keys: ";
        for (const auto &rec : *final_result_ptr)
          std::cerr << rec.key << " ";
        std::cerr << std::endl;
      }
      for (auto *c : collected_merged_chunks) {
        if (c)
          delete c;
      }
      collected_merged_chunks.clear();
      std::cerr << "[MergeCollector] SVC: Sending final EOS for pipeline."
                << std::endl;
      return EOS;
    }

    std::cerr << "[MergeCollector] SVC: Sending GO_ON (awaiting more data or "
                 "EOS from merge farm)."
              << std::endl;
    return GO_ON;
  }

private:
  void perform_k_way_merge() {
    std::cerr << "[MergeCollector] perform_k_way_merge: Starting. Collected "
              << collected_merged_chunks.size() << " chunks." << std::endl;
    if (final_result_ptr == nullptr) {
      std::cerr << "[MergeCollector] perform_k_way_merge: ERROR - "
                   "final_result_ptr is null! Cannot proceed."
                << std::endl;
      for (auto *c : collected_merged_chunks) {
        if (c)
          delete c;
      }
      collected_merged_chunks.clear();
      return;
    }

    if (collected_merged_chunks.empty()) {
      std::cerr << "[MergeCollector] perform_k_way_merge: No chunks to merge. "
                   "Clearing final result vector."
                << std::endl;
      final_result_ptr->clear();
      return;
    }

    using RecordIterator = std::vector<Record>::iterator;
    auto compare_iter_pairs =
        [](const std::pair<RecordIterator, RecordIterator> &a,
           const std::pair<RecordIterator, RecordIterator> &b) {
          if (a.first == a.second)
            return false;
          if (b.first == b.second)
            return true;
          return a.first->key > b.first->key;
        };

    std::priority_queue<std::pair<RecordIterator, RecordIterator>,
                        std::vector<std::pair<RecordIterator, RecordIterator>>,
                        decltype(compare_iter_pairs)>
        min_priority_queue(compare_iter_pairs);

    size_t total_records = 0;
    for (const auto &chunk_ptr : collected_merged_chunks) {
      if (chunk_ptr)
        total_records += chunk_ptr->data.size();
    }
    std::cerr
        << "[MergeCollector] perform_k_way_merge: Total records to merge: "
        << total_records << std::endl;

    final_result_ptr->clear();
    final_result_ptr->reserve(total_records);

    for (auto &chunk_ptr : collected_merged_chunks) {
      if (chunk_ptr && !chunk_ptr->data.empty()) {
        min_priority_queue.push(
            {chunk_ptr->data.begin(), chunk_ptr->data.end()});
      }
    }
    std::cerr << "[MergeCollector] perform_k_way_merge: Priority queue "
                 "initialized with "
              << min_priority_queue.size() << " iterators." << std::endl;

    while (!min_priority_queue.empty()) {
      std::pair<RecordIterator, RecordIterator> top_pair =
          min_priority_queue.top();
      min_priority_queue.pop();

      RecordIterator current_iter = top_pair.first;
      RecordIterator end_iter = top_pair.second;

      if (current_iter != end_iter) {
        final_result_ptr->push_back(std::move(*current_iter));
        ++current_iter;
        if (current_iter != end_iter) {
          min_priority_queue.push({current_iter, end_iter});
        }
      }
    }
    std::cerr << "[MergeCollector] perform_k_way_merge: Finished merging. "
                 "Final vector size: "
              << final_result_ptr->size() << std::endl;
  }
};

// ====== MAIN PIPELINE FUNCTION ======
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data_ref,
                                     size_t num_threads) {
  if (data_ref.empty() && num_threads > 0) {
    std::cerr << "[ff_pipeline_mergesort] Input data is empty. Exiting early."
              << std::endl;
    return;
  }
  if (num_threads == 0) {
    std::cerr << "[ff_pipeline_mergesort] num_threads is 0. Setting to 1."
              << std::endl;
    num_threads = 1;
  }

  size_t sort_workers_count;
  size_t merge_workers_count;

  if (num_threads == 1) {
    sort_workers_count = 1;
    merge_workers_count = 1;
  } else {
    sort_workers_count = std::max((size_t)1, num_threads / 2);
    merge_workers_count = std::max((size_t)1, num_threads - sort_workers_count);
  }
  std::cerr << "[ff_pipeline_mergesort] Using " << sort_workers_count
            << " sort workers and " << merge_workers_count << " merge workers."
            << std::endl;

  ff_farm sort_farm;
  std::vector<ff_node *> sort_workers_raw;
  for (size_t i = 0; i < sort_workers_count; ++i) {
    sort_workers_raw.push_back(new SortWorker());
  }
  sort_farm.add_workers(sort_workers_raw);
  sort_farm.cleanup_workers(true);

  SortEmitter *se = new SortEmitter(data_ref, sort_workers_count);
  sort_farm.add_emitter(se);
  sort_farm.cleanup_emitter(true);

  SortCollector *sc = new SortCollector(se->get_total_chunks());
  sort_farm.add_collector(sc);
  sort_farm.cleanup_collector(true);

  ff_farm merge_farm;
  std::vector<ff_node *> merge_workers_raw;
  for (size_t i = 0; i < merge_workers_count; ++i) {
    merge_workers_raw.push_back(new MergeWorker());
  }
  merge_farm.add_workers(merge_workers_raw);
  merge_farm.cleanup_workers(true);

  MergeEmitter *me = new MergeEmitter();
  merge_farm.add_emitter(me);
  merge_farm.cleanup_emitter(true);

  MergeCollector *mc = new MergeCollector(&data_ref);
  merge_farm.add_collector(mc);
  merge_farm.cleanup_collector(true);

  ff_pipeline pipeline;
  pipeline.add_stage(&sort_farm);
  pipeline.add_stage(&merge_farm);

  std::cerr << "[ff_pipeline_mergesort] Starting pipeline.run()." << std::endl;
  if (pipeline.run() < 0) {
    std::cerr << "[ff_pipeline_mergesort] ERROR: Pipeline run() failed."
              << std::endl;
    throw std::runtime_error("Pipeline run() failed");
  }
  std::cerr << "[ff_pipeline_mergesort] Calling pipeline.wait()." << std::endl;
  if (pipeline.wait() < 0) {
    std::cerr << "[ff_pipeline_mergesort] ERROR: Pipeline wait() failed."
              << std::endl;
    throw std::runtime_error("Pipeline wait() failed");
  }
  std::cerr << "[ff_pipeline_mergesort] Pipeline finished." << std::endl;
}

#ifdef TEST_MAIN
int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  std::cout << "FastFlow Pipeline Two Farms MergeSort (TEST_MAIN)\n";
  std::cout << "Array size: " << config.array_size << "\n";
  std::cout << "Payload size: " << config.payload_size << " bytes\n";
  std::cout << "Threads: " << config.num_threads << "\n\n";
  if (config.num_threads == 0) {
    std::cerr
        << "Warning: Number of threads is 0. Setting to 1 for the pipeline.\n";
    config.num_threads = 1;
  }

  auto data_for_run = generate_data_vector(config.array_size,
                                           config.payload_size, config.pattern);

  size_t original_size_for_check = data_for_run.size();

  if (config.array_size <= 20) {
    std::vector<Record> data_copy_for_debug = copy_records_vector(data_for_run);
    std::cout << "\nOriginal data (first few, if small):\n";
    for (size_t i = 0; i < std::min((size_t)10, data_copy_for_debug.size());
         ++i) {
      std::cout << "Index " << i << ": key=" << data_copy_for_debug[i].key
                << std::endl;
    }
  }

  Timer t("FF Pipeline Two Farms MergeSort");

  ff_pipeline_two_farms_mergesort(data_for_run, config.num_threads);
  double ms = t.elapsed_ms();

  std::cout << "Time: " << ms << " ms\n";

  if (config.array_size <= 20) {
    std::cout << "\nData after sort (first few, if small):\n";
    for (size_t i = 0; i < std::min((size_t)10, data_for_run.size()); ++i) {
      std::cout << "Index " << i << ": key=" << data_for_run[i].key
                << std::endl;
    }
  }

  if (data_for_run.empty() && original_size_for_check > 0) {
    std::cout << "INFO: Resulting data vector 'data_for_run' is empty."
              << std::endl;
  } else if (original_size_for_check > 0 && !data_for_run.empty()) {
    std::cout << "INFO: Resulting data vector 'data_for_run' size: "
              << data_for_run.size() << std::endl;
  } else if (original_size_for_check == 0 && data_for_run.empty()) {
    std::cout << "INFO: Original data vector 'data_for_run' was and is empty."
              << std::endl;
  }

  if (config.validate) {
    if (!is_sorted_vector(data_for_run)) {
      std::cerr << "ERROR: Sort validation failed!\n";
      if (data_for_run.size() < 200 && data_for_run.size() > 0) {
        std::cerr << "First few elements of failed sort ("
                  << data_for_run.size() << " total): ";
        for (size_t i = 0; i < std::min((size_t)20, data_for_run.size()); ++i)
          std::cerr << data_for_run[i].key << " ";
        std::cerr << std::endl;
      } else if (data_for_run.empty() && config.array_size > 0) {
        std::cerr << "Resulting vector is empty, but original size was "
                  << config.array_size << "." << std::endl;
      }
      return 1;
    } else {
      std::cout << "Validation successful.\n";
    }
  }
  return 0;
}
#endif
