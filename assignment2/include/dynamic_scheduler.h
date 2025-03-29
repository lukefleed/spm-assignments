#ifndef DYNAMIC_SCHEDULER_H
#define DYNAMIC_SCHEDULER_H

#include "common_types.h"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

// Coda thread-safe per i task
class TaskQueue {
public:
  void push(Task task);
  std::optional<Task>
  pop();        // Ritorna optional vuoto se la coda è chiusa e vuota
  void close(); // Segnala che non verranno aggiunti altri task

private:
  std::queue<Task> queue_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  bool closed_ = false;
};

/**
 * @brief Esegue il calcolo usando scheduling dinamico con una task queue.
 * @param config Configurazione del programma.
 * @param results_out Vettore (inizializzato) dove salvare i risultati
 * (RangeResult).
 * @return true se successo, false altrimenti.
 */
bool run_dynamic_task_queue(const Config &config,
                            std::vector<RangeResult> &results_out);

#endif // DYNAMIC_SCHEDULER_H
