//// filepath:
////home/lukefleed/MEGA/Università/magistrale-CS/SPM/24-25/spm-assignments/assignment2/src/dynamic_scheduler.cpp
#include "dynamic_scheduler.h"
#include "collatz.h"
#include <atomic>
#include <iostream> // Per debug/verbose
#include <limits>
#include <optional>
#include <thread>
#include <vector>

// --- Implementazione TaskQueue ---

void TaskQueue::push(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      // O lancia eccezione o ignora silenziosamente
      // throw std::runtime_error("Cannot push to a closed queue");
      return;
    }
    queue_.push(std::move(task));
  }                       // Lock rilasciato
  cond_var_.notify_one(); // Notifica un thread in attesa
}

std::optional<Task> TaskQueue::pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  // Attendi finché la coda non è vuota O è chiusa
  cond_var_.wait(lock, [this] { return !queue_.empty() || closed_; });

  // Se la coda è vuota e chiusa, termina
  if (queue_.empty() && closed_) {
    return std::nullopt;
  }
  if (queue_.empty()) {
    return std::nullopt;
  }
  Task task = std::move(queue_.front());
  queue_.pop();
  return task;
}

void TaskQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  cond_var_.notify_all(); // Sveglia tutti i thread in attesa
}

// --- Implementazione Scheduler Dinamico ---

// Funzione eseguita da ogni worker thread nello scheduler dinamico
void dynamic_worker(int thread_id,    // Per debugging
                    TaskQueue &queue, // Riferimento alla coda condivisa
                    std::vector<RangeResult> &
                        results_out) { // Riferimento per aggiornare i risultati
  while (true) {
    std::optional<Task> task_opt = queue.pop();
    if (!task_opt) {
      break;
    }
    Task task = *task_opt; // Estrai il task

    ull local_max_steps = find_max_steps_in_subrange(task.start, task.end);

    // Aggiorna il massimo per il range corrispondente in modo atomico
    ull current_max = results_out[task.original_range_index].max_steps.load(
        std::memory_order_relaxed);
    while (local_max_steps > current_max) {
      if (results_out[task.original_range_index]
              .max_steps.compare_exchange_weak(current_max, local_max_steps,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
        break;
      }
      std::this_thread::yield();
    }
  }
}

bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out) {
  if (config.num_threads <= 0 || config.chunk_size == 0)
    return false;

  TaskQueue task_queue;
  std::vector<std::thread> threads;

  results_out.clear();
  // Inizializza il vettore dei risultati
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r);
  }

  // Crea e avvia i worker thread
  for (int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(dynamic_worker, i, std::ref(task_queue),
                         std::ref(results_out));
  }

  // Il thread main popola la coda con i task
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    const auto &range = config.ranges[i];
    ull current_start = range.start;
    while (current_start <= range.end) {
      ull current_chunk_end;
      // Fix: Guard against unsigned long long overflow when computing the end
      // of the chunk
      if (current_start >
          std::numeric_limits<ull>::max() - (config.chunk_size - 1)) {
        current_chunk_end = range.end;
      } else {
        current_chunk_end = current_start + config.chunk_size - 1;
      }
      ull current_end = std::min(range.end, current_chunk_end);
      task_queue.push({current_start, current_end, static_cast<int>(i)});

      // Se abbiamo raggiunto la fine del range, esci dal ciclo
      if (current_end == range.end) {
        break;
      }
      current_start = current_end + 1;
    }
  }

  // Segnala ai thread che non ci sono più task
  task_queue.close();

  // Attendi il completamento di tutti i worker thread
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  return true;
}
