#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common_types.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "testing.h" // Includi il nuovo header
#include "utils.h"

bool run_sequential_wrapper(const Config &config,
                            std::vector<RangeResult> &results_out);

int main(int argc, char *argv[]) {

  // --- Gestione Modalità Test ---
  // Controlla se è stata richiesta una modalità di test prima del parsing
  // normale
  if (argc >= 2) {
    std::string first_arg = argv[1];
    if (first_arg == "--test-correctness") {
      bool success = run_correctness_suite();
      return success ? 0 : 1; // Esce dopo i test di correttezza
    } else if (first_arg == "--test-performance") {
      // --- Parametri per i test di performance ---
      // Workload più leggero
      std::vector<Range> perf_workload = {{1, 1000}, {1000000, 2000000}};
      // Potrebbe essere meglio un workload più pesante per vedere meglio lo
      // scaling std::vector<Range> perf_workload = {{1, 200000000}}; // Esempio
      // alternativo

      // Thread da testare
      std::vector<int> threads_to_test = {
          1, 2, 4, 8, 12}; // Adatta ai core della macchina di test
      // Chunk/Block size da testare (ridotti)
      std::vector<ull> chunks_to_test = {128}; // Ridotto da {64, 256}

      int samples = 5;               // Numero di misurazioni mediane
      int iterations_per_sample = 3; // Esecuzioni per ogni misurazione

      // --- Fine Parametri ---

      bool success =
          run_performance_suite(threads_to_test, chunks_to_test, samples,
                                iterations_per_sample, perf_workload);
      // L'output CSV va a stdout, verrà rediretto dal Makefile.
      return success ? 0 : 1; // Esce dopo i test di performance
    }
    // Se non è un flag di test, continua con il parsing normale sotto
  }

  // --- Esecuzione Normale (Calcolo Collatz) ---

  // 1. Parsing Argomenti (come prima)
  auto config_opt = parse_arguments(argc, argv);
  if (!config_opt) {
    return 1;
  }
  Config config = *config_opt;

  std::vector<RangeResult> results;

  // 2. Selezione ed Esecuzione Scheduler (come prima)
  Timer timer;
  bool success = false;

  // Modificato leggermente per usare la wrapper per il caso sequenziale puro
  if (config.num_threads == 1) {
    if (config.verbose) {
      std::cout << "Running sequential version (num_threads=1 specified)..."
                << std::endl;
    }
    timer.reset();
    success = run_sequential_wrapper(config, results); // Usa la wrapper
  } else if (config.scheduling == SchedulingType::STATIC) {
    if (config.verbose) {
      std::cout << "Running static block-cyclic scheduler..." << std::endl;
    }
    timer.reset();
    success = run_static_block_cyclic(config, results);
  } else { // SchedulingType::DYNAMIC
    if (config.verbose) {
      std::cout << "Running dynamic task queue scheduler..." << std::endl;
    }
    timer.reset();
    success = run_dynamic_task_queue(config, results);
  }

  double elapsed_time_s = timer.elapsed_s();

  // 3. Stampa Risultati (come prima)
  if (success) {
    // Stampa nel formato richiesto standard
    for (const auto &res : results) {
      std::cout << res.original_range.start << "-" << res.original_range.end
                << ": " << res.max_steps.load() << std::endl;
    }

    // Stampa informazioni aggiuntive se verbose (già presente)
    if (config.verbose) {
      std::cout << "\nTotal execution time: " << std::fixed
                << std::setprecision(4) << elapsed_time_s << " seconds"
                << std::endl;
      std::cout << "Using " << config.num_threads << " threads." << std::endl;
      if (config.scheduling == SchedulingType::STATIC &&
          config.num_threads > 1) { // Aggiunto controllo > 1
        std::cout << "Scheduling: Static Block-Cyclic, Block Size: "
                  << config.chunk_size << std::endl;
      } else if (config.scheduling ==
                 SchedulingType::DYNAMIC) { // Il dinamico ha senso anche con 1
                                            // thread (ma inutile)
        std::cout << "Scheduling: Dynamic Task Queue, Chunk Size: "
                  << config.chunk_size << std::endl;
      } else { // Caso sequenziale
        std::cout << "Scheduling: Sequential" << std::endl;
      }
    }
  } else {
    std::cerr << "Error during computation." << std::endl;
    return 1;
  }

  return 0;
}
