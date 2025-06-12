#include "ff_mergesort.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
#include <vector>

using namespace ff;

namespace {

// Represents the state of the sorting process, passed through the pipeline.
struct MergeSortState {
  Record *source;
  Record *dest;
  const size_t n;
  const size_t num_threads;
  const size_t chunk_size;
  size_t merge_width;

  MergeSortState(Record *s, Record *d, size_t size, size_t threads)
      : source(s), dest(d), n(size), num_threads(threads),
        chunk_size(std::max(static_cast<size_t>(1024),
                            n / (threads > 0 ? (threads * 4) : 4))),
        merge_width(0) {}
};

// Represents a task for either sorting or merging a sub-array.
struct Task {
  Record *source;
  Record *dest;
  size_t start;
  size_t mid;
  size_t end;
};

// First stage of the pipeline: sorts initial chunks in parallel.
class InitialSorterNode : public ff_node_t<MergeSortState, MergeSortState> {
private:
  const size_t num_threads;

public:
  InitialSorterNode(size_t nw) : num_threads(nw) {}

  MergeSortState *svc(MergeSortState *state) override {
    struct Emitter : ff_node {
      MergeSortState *s;
      size_t offset = 0;
      Emitter(MergeSortState *state) : s(state) {}
      void *svc(void *) override {
        if (offset >= s->n)
          return EOS;
        size_t start = offset;
        size_t end = std::min(start + s->chunk_size, s->n);
        offset = end;
        return new Task{s->source, nullptr, start, 0, end};
      }
    };

    struct Worker : ff_node_t<Task, void> {
      void *svc(Task *task) override {
        std::sort(task->source + task->start, task->source + task->end);
        delete task;
        return GO_ON;
      }
    };

    ff_farm sort_farm;
    sort_farm.add_emitter(new Emitter(state));
    sort_farm.cleanup_emitter(true);
    std::vector<ff_node *> workers;
    workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers.push_back(new Worker());
    }
    sort_farm.add_workers(workers);
    sort_farm.cleanup_workers(true);

    if (sort_farm.run_and_wait_end() < 0) {
      // FastFlow farm failed - return EOS to terminate pipeline
      return this->EOS;
    }

    state->merge_width = state->chunk_size;
    return state;
  }
};

// Second stage of the pipeline: iteratively merges sorted chunks.
class MergeNode : public ff_node_t<MergeSortState, MergeSortState> {
private:
  const size_t num_threads;

public:
  MergeNode(size_t nw) : num_threads(nw) {}

  MergeSortState *svc(MergeSortState *state) override {
    struct Emitter : ff_node {
      MergeSortState *s;
      size_t offset = 0;
      Emitter(MergeSortState *state) : s(state) {}
      void *svc(void *) override {
        if (offset >= s->n)
          return EOS;
        size_t start = offset;
        size_t mid = std::min(start + s->merge_width, s->n);
        size_t end = std::min(start + 2 * s->merge_width, s->n);
        offset = end;
        return new Task{s->source, s->dest, start, mid, end};
      }
    };

    struct Worker : ff_node_t<Task, void> {
      void *svc(Task *task) override {
        if (task->mid >= task->end) {
          std::move(task->source + task->start, task->source + task->mid,
                    task->dest + task->start);
        } else {
          std::merge(std::make_move_iterator(task->source + task->start),
                     std::make_move_iterator(task->source + task->mid),
                     std::make_move_iterator(task->source + task->mid),
                     std::make_move_iterator(task->source + task->end),
                     task->dest + task->start);
        }
        delete task;
        return GO_ON;
      }
    };

    while (state->merge_width < state->n) {
      ff_farm merge_farm;
      merge_farm.add_emitter(new Emitter(state));
      merge_farm.cleanup_emitter(true);
      std::vector<ff_node *> workers;
      workers.reserve(num_threads);
      for (size_t i = 0; i < num_threads; ++i) {
        workers.push_back(new Worker());
      }
      merge_farm.add_workers(workers);
      merge_farm.cleanup_workers(true);

      if (merge_farm.run_and_wait_end() < 0) {
        return this->EOS;
      }

      std::swap(state->source, state->dest);
      state->merge_width *= 2;
    }
    return state;
  }
};

// A simple node to inject the single initial task into the pipeline.
class TaskFeeder : public ff_node_t<MergeSortState, MergeSortState> {
private:
  MergeSortState *const task;
  bool sent = false;

public:
  TaskFeeder(MergeSortState *t) : task(t) {}
  MergeSortState *svc(MergeSortState *) override {
    if (!sent) {
      sent = true;
      return task;
    }
    return this->EOS; // Send End-Of-Stream after the first task
  }
};

} // namespace

void parallel_mergesort(std::vector<Record> &data, const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1) {
    return;
  }

  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  std::vector<Record> aux_buffer(n);
  auto state = std::make_unique<MergeSortState>(data.data(), aux_buffer.data(),
                                                n, effective_threads);

  TaskFeeder feeder(state.get());
  InitialSorterNode sorter(effective_threads);
  MergeNode merger(effective_threads);

  ff_pipeline pipeline;
  // The feeder becomes the first stage
  pipeline.add_stage(&feeder);
  pipeline.add_stage(&sorter);
  pipeline.add_stage(&merger);

  if (pipeline.run_and_wait_end() < 0) {
    throw std::runtime_error("Parallel mergesort pipeline failed");
  }

  // Check if the final sorted data is in the auxiliary buffer.
  // The final source pointer is in state->source after the last swap in
  // MergeNode.
  if (state->source != data.data()) {
    std::move(state->source, state->source + n, data.data());
  }
}
