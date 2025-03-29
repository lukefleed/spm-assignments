#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h"    // Per Timer
#include <algorithm>  // Per std::sort, std::minmax_element
#include <cmath>      // Per std::sqrt, std::abs
#include <functional> // Per std::function
#include <iomanip>    // Per std::setw, std::fixed, std::setprecision
#include <iostream>
#include <numeric> // Per std::accumulate
#include <vector>

// --- Test di Correttezza ---

struct CorrectnessTestCase {
  std::string name;
  std::vector<Range> ranges;
  std::vector<int> thread_counts;
  std::vector<ull> chunk_sizes;
};

bool run_correctness_suite() {
  std::cout << "--- Running Correctness Suite ---" << std::endl;
  bool all_passed = true;

  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {1, 2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {1, 2}, {1}},
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {4, 8},
       {1, 10}},
      {"Larger Range", {{1, 10000}}, {8, 16}, {64, 128}},
      // Aggiungere altri casi se necessario (es. range con start>end già
      // gestito nel parsing, range con 0)
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {4}, {16}}};

  int test_count = 0;
  int failed_count = 0;

  for (const auto &tc : test_cases) {
    test_count++;
    std::cout << "\n[Test Case " << test_count << ": " << tc.name << "]"
              << std::endl;
    bool case_passed = true;

    // 1. Esegui Sequenziale (Baseline)
    std::cout << "  Running Sequential..." << std::flush;
    std::vector<ull> expected_results = run_sequential(tc.ranges);
    std::cout << " Done." << std::endl;

    // 2. Esegui Statico e Dinamico con varie configurazioni
    for (int n_threads : tc.thread_counts) {
      for (ull chunk : tc.chunk_sizes) {
        Config config_static;
        config_static.scheduling = SchedulingType::STATIC;
        config_static.num_threads = n_threads;
        config_static.chunk_size = chunk;
        config_static.ranges = tc.ranges;
        config_static.verbose = false; // Tenere spento per output pulito

        Config config_dynamic = config_static; // Copia la base
        config_dynamic.scheduling = SchedulingType::DYNAMIC;

        // --- Static Test ---
        std::cout << "  Testing Static  (T=" << n_threads << ", C=" << chunk
                  << ")..." << std::flush;
        std::vector<RangeResult> static_results_rr;
        if (run_static_block_cyclic(config_static, static_results_rr)) {
          bool match = true;
          if (static_results_rr.size() != expected_results.size()) {
            match = false;
          } else {
            for (size_t i = 0; i < expected_results.size(); ++i) {
              if (static_results_rr[i].max_steps.load() !=
                  expected_results[i]) {
                match = false;
                break;
              }
            }
          }
          if (match) {
            std::cout << " PASS" << std::endl;
          } else {
            std::cout << " FAIL (Mismatch)" << std::endl;
            // Qui potresti stampare i risultati attesi vs ottenuti per debug
            case_passed = false;
          }
        } else {
          std::cout << " FAIL (Execution Error)" << std::endl;
          case_passed = false;
        }

        // --- Dynamic Test ---
        std::cout << "  Testing Dynamic (T=" << n_threads << ", C=" << chunk
                  << ")..." << std::flush;
        std::vector<RangeResult> dynamic_results_rr;
        if (run_dynamic_task_queue(config_dynamic, dynamic_results_rr)) {
          bool match = true;
          if (dynamic_results_rr.size() != expected_results.size()) {
            match = false;
          } else {
            for (size_t i = 0; i < expected_results.size(); ++i) {
              if (dynamic_results_rr[i].max_steps.load() !=
                  expected_results[i]) {
                match = false;
                break;
              }
            }
          }
          if (match) {
            std::cout << " PASS" << std::endl;
          } else {
            std::cout << " FAIL (Mismatch)" << std::endl;
            case_passed = false;
          }
        } else {
          std::cout << " FAIL (Execution Error)" << std::endl;
          case_passed = false;
        }
      }
    }

    if (!case_passed) {
      failed_count++;
      all_passed = false;
    }
  }

  std::cout << "\n--- Correctness Suite Summary ---" << std::endl;
  std::cout << "Total Test Cases: " << test_count << std::endl;
  std::cout << "Passed: " << test_count - failed_count << std::endl;
  std::cout << "Failed: " << failed_count << std::endl;
  std::cout << "---------------------------------" << std::endl;

  return all_passed;
}

// --- Test di Performance ---

// Funzione helper generica per misurare il tempo di esecuzione (mediana)
// Accetta una funzione che esegue il calcolo e restituisce bool (success/fail)
double measure_median_time_ms(
    std::function<bool(const Config &, std::vector<RangeResult> &)> func_to_run,
    const Config &config, int samples, int iterations_per_sample) {
  if (samples <= 0 || iterations_per_sample <= 0)
    return -1.0; // Valori non validi

  std::vector<double> all_iteration_times_ms;
  all_iteration_times_ms.reserve(samples * iterations_per_sample);

  std::vector<RangeResult>
      results_buffer; // Buffer per i risultati (non analizzati qui)

  for (int s = 0; s < samples; ++s) {
    for (int iter = 0; iter < iterations_per_sample; ++iter) {
      Timer timer;
      bool success = func_to_run(config, results_buffer);
      double duration_ms = timer.elapsed_ms();

      if (!success) {
        std::cerr << "Warning: Execution failed during performance measurement "
                     "for config (T="
                  << config.num_threads << ", C=" << config.chunk_size
                  << ", Sched="
                  << (config.scheduling == SchedulingType::STATIC ? "Static"
                                                                  : "Dynamic")
                  << "). Sample " << s << ", Iter " << iter
                  << ". Skipping time." << std::endl;
        // Potresti decidere di invalidare l'intero campione o il test
        continue; // Salta questo tempo
      }
      all_iteration_times_ms.push_back(duration_ms);
    }
  }

  if (all_iteration_times_ms.empty()) {
    return -2.0; // Nessuna misurazione valida
  }

  // Calcola la mediana
  std::sort(all_iteration_times_ms.begin(), all_iteration_times_ms.end());
  size_t n = all_iteration_times_ms.size();
  if (n % 2 != 0) {
    return all_iteration_times_ms[n / 2];
  } else {
    return (all_iteration_times_ms[n / 2 - 1] + all_iteration_times_ms[n / 2]) /
           2.0;
  }
}

// Funzione wrapper per run_sequential per adattarla a std::function
bool run_sequential_wrapper(const Config &config,
                            std::vector<RangeResult> &results_out) {
  auto seq_max_steps = run_sequential(config.ranges);
  results_out.clear();
  results_out.reserve(config.ranges.size());
  for (size_t i = 0; i < config.ranges.size(); ++i) {
    // Dobbiamo creare RangeResult anche qui per coerenza interfaccia
    results_out.emplace_back(config.ranges[i]);
    results_out.back().max_steps.store(seq_max_steps[i]);
  }
  return true; // run_sequential non ritorna bool, assumiamo successo
}

bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload) {
  std::cout << "--- Running Performance Suite ---" << std::endl;
  std::cout << "Samples per config: " << samples
            << ", Iterations per sample: " << iterations_per_sample
            << std::endl;
  std::cout << "Workload Ranges: ";
  for (const auto &r : workload)
    std::cout << r.start << "-" << r.end << " ";
  std::cout << std::endl;

  // Stampa header CSV
  std::cout << "\nScheduler,Threads,ChunkSize,MedianTimeMs" << std::endl;

  Config base_config;
  base_config.ranges = workload;
  base_config.verbose = false;

  // 1. Test Sequenziale (Baseline)
  Config seq_config = base_config;
  seq_config.num_threads =
      1; // Anche se non usato da run_sequential, per coerenza
  seq_config.chunk_size = 0; // Non applicabile
  seq_config.scheduling =
      SchedulingType::STATIC; // Non rilevante ma deve essere qualcosa

  // Creiamo la funzione da passare (lambda che chiama la wrapper)
  auto seq_func = [](const Config &cfg, std::vector<RangeResult> &res) {
    return run_sequential_wrapper(cfg, res);
  };

  double median_seq_ms = measure_median_time_ms(seq_func, seq_config, samples,
                                                iterations_per_sample);
  if (median_seq_ms >= 0) {
    std::cout << "Sequential," << 1 << ","
              << "N/A"
              << "," << std::fixed << std::setprecision(4) << median_seq_ms
              << std::endl;
  } else {
    std::cout << "Sequential," << 1 << ","
              << "N/A"
              << ","
              << "ERROR" << std::endl;
    // Potremmo voler interrompere se il sequenziale fallisce, ma continuiamo
    // per ora
  }

  // 2. Test Statico e Dinamico
  for (int n_threads : thread_counts) {
    // Se n_threads è 1, potremmo saltare per evitare duplicati col test
    // sequenziale, ma misurarlo può essere utile per vedere l'overhead della
    // struttura parallela. Lo misuriamo comunque.

    for (ull chunk : chunk_sizes) {
      // --- Static Test ---
      Config config_static = base_config;
      config_static.scheduling = SchedulingType::STATIC;
      config_static.num_threads = n_threads;
      config_static.chunk_size = chunk;

      // Creiamo la funzione da passare
      auto static_func = [](const Config &cfg, std::vector<RangeResult> &res) {
        return run_static_block_cyclic(cfg, res);
      };

      double median_static_ms = measure_median_time_ms(
          static_func, config_static, samples, iterations_per_sample);

      if (median_static_ms >= 0) {
        std::cout << "Static," << n_threads << "," << chunk << "," << std::fixed
                  << std::setprecision(4) << median_static_ms << std::endl;
      } else {
        std::cout << "Static," << n_threads << "," << chunk << ","
                  << "ERROR" << std::endl;
      }

      // --- Dynamic Test ---
      Config config_dynamic = base_config;
      config_dynamic.scheduling = SchedulingType::DYNAMIC;
      config_dynamic.num_threads = n_threads;
      config_dynamic.chunk_size = chunk;

      // Creiamo la funzione da passare
      auto dynamic_func = [](const Config &cfg, std::vector<RangeResult> &res) {
        return run_dynamic_task_queue(cfg, res);
      };

      double median_dynamic_ms = measure_median_time_ms(
          dynamic_func, config_dynamic, samples, iterations_per_sample);

      if (median_dynamic_ms >= 0) {
        std::cout << "Dynamic," << n_threads << "," << chunk << ","
                  << std::fixed << std::setprecision(4) << median_dynamic_ms
                  << std::endl;
      } else {
        std::cout << "Dynamic," << n_threads << "," << chunk << ","
                  << "ERROR" << std::endl;
      }
    }
  }

  std::cout << "\n--- Performance Suite Finished ---" << std::endl;
  return true; // Indica che la suite è stata eseguita (non che i tempi sono
               // buoni o i calcoli corretti)
}
