#include "ff_mergesort.hpp"
#include "../common/record.hpp"
#include <algorithm>
#include <ff/ff.hpp>
#include <memory>
#include <vector>

using namespace ff;

namespace {

/**
 * @struct MergeTask
 * @brief Definisce un'operazione di sort o merge. Non possiede memoria,
 *        prevenendo l'overhead di allocazione durante l'esecuzione.
 */
struct MergeTask {
  Record *source;
  Record *dest;
  size_t start;
  size_t mid;
  size_t end;
};

/**
 * @class Emitter
 * @brief Emette task per una passata di sort o merge.
 */
class Emitter : public ff_node {
public:
  Emitter(size_t total_size, size_t step, Record *from_buf,
          Record *to_buf = nullptr)
      : n(total_size), step_size(step), from(from_buf), to(to_buf), offset(0) {}

  void *svc(void *) override {
    if (offset >= n) {
      return EOS;
    }

    size_t start = offset;
    size_t mid = std::min(start + step_size, n);
    size_t end = std::min(start + 2 * step_size, n);

    // Per la fase di sort iniziale (quando 'to' è nullo), il task definisce
    // solo un segmento da ordinare in-place, identificato da start e end (che è
    // uguale a mid).
    if (to == nullptr) {
      end = mid;
    }

    auto *task = new MergeTask{from, to, start, mid, end};
    offset = end;

    return task;
  }

private:
  const size_t n;
  const size_t step_size;
  Record *const from;
  Record *const to;
  size_t offset;
};

/**
 * @class SortWorker
 * @brief Ordina una regione specificata di un buffer in-place.
 */
class SortWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::sort(task->source + task->start, task->source + task->end);
    delete task;
    return GO_ON;
  }
};

/**
 * @class MergeWorker
 * @brief Fonde due sotto-regioni ordinate da un buffer sorgente a uno
 * destinazione.
 */
class MergeWorker : public ff_node_t<MergeTask, void> {
public:
  void *svc(MergeTask *task) override {
    std::merge(std::make_move_iterator(task->source + task->start),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->mid),
               std::make_move_iterator(task->source + task->end),
               task->dest + task->start);
    delete task;
    return GO_ON;
  }
};

} // anonymous namespace

/**
 * @brief Sorts a vector of Records using a highly performant and correct
 * parallel merge sort implementation based on synchronized merge passes.
 * @note The function name is kept for API compatibility. The implementation
 * does not use a pipeline of farms but a more robust sequence of synchronized
 * farms.
 */
void ff_pipeline_two_farms_mergesort(std::vector<Record> &data,
                                     const size_t num_threads) {
  const size_t n = data.size();
  if (n <= 1)
    return;

  const size_t effective_threads = (num_threads == 0) ? 1 : num_threads;

  if (n < effective_threads * 1024) {
    std::sort(data.begin(), data.end());
    return;
  }

  // --- Fase 1: Sort Parallelo In-place dei Chunk Iniziali ---
  const size_t chunk_size =
      std::max(static_cast<size_t>(1024), n / (effective_threads * 4));

  ff_farm sort_farm;
  sort_farm.add_emitter(new Emitter(n, chunk_size, data.data()));
  sort_farm.cleanup_emitter(true);

  std::vector<ff_node *> sorters;
  sorters.reserve(effective_threads);
  for (size_t i = 0; i < effective_threads; ++i) {
    sorters.push_back(new SortWorker());
  }
  sort_farm.add_workers(sorters);
  sort_farm.cleanup_workers(true);

  if (sort_farm.run_and_wait_end() < 0) {
    throw std::runtime_error("Initial sorting farm failed");
  }

  // --- Fase 2: Merge Parallelo a Passate Sincronizzate ---
  std::vector<Record> aux_buffer(n);
  Record *from = data.data();
  Record *to = aux_buffer.data();

  for (size_t width = chunk_size; width < n; width *= 2) {
    ff_farm merge_farm;
    merge_farm.add_emitter(new Emitter(n, width, from, to));
    merge_farm.cleanup_emitter(true);

    std::vector<ff_node *> mergers;
    mergers.reserve(effective_threads);
    for (size_t i = 0; i < effective_threads; ++i) {
      mergers.push_back(new MergeWorker());
    }
    merge_farm.add_workers(mergers);
    merge_farm.cleanup_workers(true);

    if (merge_farm.run_and_wait_end() < 0) {
      throw std::runtime_error("Merge farm failed");
    }

    std::swap(from, to);
  }

  // --- Fase 3: Spostamento Finale dei Dati Ordinati ---
  if (from != data.data()) {
    std::move(from, from + n, data.data());
  }
}
