#include "static_scheduler.h"
#include "collatz.h"
#include <atomic>
#include <cmath>    // Per std::ceil
#include <iostream> // Per debug/verbose
#include <thread>
#include <vector>

// Funzione eseguita da ogni thread nello scheduler statico
void static_worker(
    int thread_id, int num_threads, ull block_size,
    const std::vector<Range>
        &global_ranges, // Copia o reference const? Reference const va bene
    std::vector<RangeResult> &results // Riferimento non const per aggiornare
                                      // bool verbose // Opzionale per debug
) {
  // Calcolo "globale" dei blocchi
  ull current_global_block_index = 0; // Indice del blocco corrente rispetto a
                                      // tutti i numeri in tutti i range
  ull total_numbers_processed_in_prev_ranges = 0;

  if (block_size == 0) {
    // Avoid division by zero if block_size is invalid (should be caught
    // earlier)
    return;
  }

  for (size_t range_idx = 0; range_idx < global_ranges.size(); ++range_idx) {
    const auto &current_range = global_ranges[range_idx];
    ull range_len = current_range.end - current_range.start + 1;
    ull num_blocks_in_range =
        (range_len + block_size - 1) / block_size; // Ceiling division

    // Itera sui blocchi di questo range
    for (ull block_in_range_idx = 0; block_in_range_idx < num_blocks_in_range;
         ++block_in_range_idx) {
      ull global_block_idx =
          total_numbers_processed_in_prev_ranges / block_size +
          block_in_range_idx;

      // Assegnazione block-cyclic
      if (global_block_idx % num_threads == thread_id) {
        // Questo thread processa questo blocco
        ull block_start = current_range.start + block_in_range_idx * block_size;
        ull block_end =
            std::min(current_range.end, block_start + block_size - 1);

        // if (verbose) {
        //     std::cout << "Thread " << thread_id << " processing block " <<
        //     global_block_idx
        //               << " (Range " << range_idx << ", local block " <<
        //               block_in_range_idx
        //               << ", [" << block_start << "-" << block_end << "])" <<
        //               std::endl;
        // }

        ull local_max_steps =
            find_max_steps_in_subrange(block_start, block_end);

        // Aggiorna il massimo per il range originale in modo atomico
        // fetch_max è C++20, usiamo compare_exchange_weak/strong per C++17
        ull current_max =
            results[range_idx].max_steps.load(std::memory_order_relaxed);
        while (local_max_steps > current_max) {
          if (results[range_idx].max_steps.compare_exchange_weak(
                  current_max, local_max_steps, std::memory_order_release,
                  std::memory_order_relaxed)) {
            break; // Success
          }
          // Failure (current_max è stato aggiornato), riprova con il nuovo
          // valore
        }
        // Alternativa C++20:
        // results[range_idx].max_steps.fetch_max(local_max_steps,
        // std::memory_order_relaxed);
      }
    }
    total_numbers_processed_in_prev_ranges +=
        range_len; // Aggiorna per il prossimo range
  }
  // if (verbose) std::cout << "Thread " << thread_id << " finished." <<
  // std::endl;
}

bool run_static_block_cyclic(const Config &config,
                             std::vector<RangeResult> &results_out) {
  if (config.num_threads <= 0 || config.chunk_size == 0)
    return false; // Input non validi

  std::vector<std::thread> threads;
  results_out.clear(); // Assicurati che il vettore sia vuoto
  // Inizializza il vettore dei risultati (uno per ogni range di input)
  for (const auto &r : config.ranges) {
    results_out.emplace_back(r); // Usa il costruttore di RangeResult(Range)
  }

  // if (config.verbose) {
  //     std::cout << "Starting static block-cyclic with " << config.num_threads
  //               << " threads and block size " << config.chunk_size <<
  //               std::endl;
  // }

  // Crea e avvia i thread
  for (int i = 0; i < config.num_threads; ++i) {
    threads.emplace_back(static_worker, i, config.num_threads,
                         config.chunk_size,
                         std::cref(config.ranges), // Passa per reference const
                         std::ref(results_out) // Passa per reference non-const
                         // config.verbose
    );
  }

  // Attendi il completamento di tutti i thread
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // if (config.verbose) {
  //     std::cout << "Static block-cyclic finished." << std::endl;
  // }

  return true;
}
