#include "common_types.h"
#include "dynamic_scheduler.h" // Potrebbe non essere più necessario qui se non richiamato direttamente
#include "sequential.h" // Potrebbe non essere più necessario qui se non richiamato direttamente
#include "static_scheduler.h" // Potrebbe non essere più necessario qui se non richiamato direttamente
#include "testing.h"
#include "utils.h"
#include <chrono>
#include <cstdlib> // For EXIT_SUCCESS, EXIT_FAILURE
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view> // Use string_view for comparisons
#include <thread>
#include <vector>

// --- Constants for Command Line Arguments ---
namespace AppConstants {
constexpr const char *TEST_CORRECTNESS_FLAG = "--test-correctness";
constexpr const char *BENCHMARK_FLAG = "--benchmark"; // Nuovo flag unificato
} // namespace AppConstants

/**
 * @brief Get the scheduler type name as string (usato solo per output normale)
 * @param config Application configuration
 * @return std::string Descriptive name of scheduler
 */
std::string get_scheduler_name(const Config &config) {
  if (config.num_threads == 1) {
    return "Sequential";
  } else if (config.scheduling == SchedulingType::STATIC) {
    std::string variant_name;
    switch (config.static_variant) {
    case StaticVariant::BLOCK:
      variant_name = "Block";
      break;
    case StaticVariant::CYCLIC:
      variant_name = "Cyclic";
      break;
    case StaticVariant::BLOCK_CYCLIC:
      variant_name = "Block-Cyclic";
      break;
    default:
      variant_name = "Unknown";
      break; // Robustezza
    }
    return "Static " + variant_name;
  } else if (config.scheduling == SchedulingType::DYNAMIC) {
    return "Dynamic Task Queue";
  } else {
    return "Unknown Scheduler Type";
  }
}

/**
 * @brief Execute the appropriate Collatz calculation based on configuration
 * (usato solo per esecuzione normale)
 * @param config Application configuration
 * @param results Output vector for calculation results
 * @return true on successful execution, false otherwise
 */
bool execute_collatz_calculation(const Config &config,
                                 std::vector<RangeResult> &results) {
  std::string scheduler_name = get_scheduler_name(config);
  bool success = false;

  if (config.verbose) {
    std::cout << "Running " << scheduler_name << " scheduler..." << std::endl;
  }

  try {
    // NOTA: La logica di scelta dello scheduler è ora centralizzata qui.
    // run_sequential_wrapper non serve più in main.
    if (config.num_threads == 1) {
      // Usa direttamente run_sequential e adatta il risultato
      std::vector<ull> seq_results = run_sequential(config.ranges);
      results.clear();
      results.reserve(seq_results.size());
      for (size_t i = 0; i < seq_results.size(); ++i) {
        results.emplace_back(config.ranges[i]);
        results.back().max_steps.store(seq_results[i]);
      }
      success = true;
    } else if (config.scheduling == SchedulingType::STATIC) {
      success = run_static_scheduling(config, results);
    } else if (config.scheduling == SchedulingType::DYNAMIC) {
      success = run_dynamic_task_queue(config, results);
    } else {
      std::cerr << "Error: Unknown scheduling type configured." << std::endl;
      success = false;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during " << scheduler_name << " execution: " << e.what()
              << std::endl;
    success = false;
  } catch (...) {
    std::cerr << "Unknown error during " << scheduler_name << " execution."
              << std::endl;
    success = false;
  }
  return success;
}

/**
 * @brief Print calculation results and execution statistics
 * (usato solo per esecuzione normale)
 */
void print_results(const std::vector<RangeResult> &results,
                   const Config &config, double elapsed_time_s) {
  // Print calculation results
  for (const auto &res : results) {
    std::cout << res.original_range.start << "-" << res.original_range.end
              << ": " << res.max_steps.load(std::memory_order_relaxed)
              << std::endl;
  }

  // Print performance statistics if verbose
  if (config.verbose) {
    std::string scheduler_name = get_scheduler_name(config);
    std::cout << "\n--- Execution Summary ---" << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(4)
              << elapsed_time_s << " seconds" << std::endl;
    std::cout << "Threads used: " << config.num_threads << std::endl;
    std::cout << "Scheduling: " << scheduler_name;
    if (config.num_threads > 1 &&
        (config.scheduling == SchedulingType::STATIC ||
         config.scheduling == SchedulingType::DYNAMIC)) {
      std::cout << ", Chunk Size: " << config.chunk_size;
    }
    std::cout << std::endl;
    std::cout << "------------------------" << std::endl;
  }
}

/**
 * @brief Checks if the first command-line argument is a known test/benchmark
 * flag.
 */
[[nodiscard]] bool is_test_or_benchmark_mode(int argc, char *argv[]) {
  if (argc < 2) {
    return false;
  }
  const std::string_view first_arg(argv[1]);
  return first_arg == AppConstants::TEST_CORRECTNESS_FLAG ||
         first_arg == AppConstants::BENCHMARK_FLAG;
}

/**
 * @brief Execute test suites or benchmarks based on command line arguments
 */
[[nodiscard]] bool handle_test_or_benchmark_mode(int argc, char *argv[]) {
  if (argc < 2)
    return false;

  const std::string_view first_arg(argv[1]);

  if (first_arg == AppConstants::TEST_CORRECTNESS_FLAG) {
    std::cout << "Running Correctness Test Suite..." << std::endl;
    return run_correctness_suite();
  }

  if (first_arg == AppConstants::BENCHMARK_FLAG) {
    std::cout << "Running Performance Benchmark Suite..." << std::endl;
    // --- Configurazione del Benchmark ---
    // Puoi rendere questi configurabili tramite altri argomenti se necessario

    // Thread da testare (fino al massimo hardware)
    const int max_threads = std::thread::hardware_concurrency();
    std::vector<int> threads_to_test;
    // Inizia da 2 per i test paralleli, 1 è il baseline sequenziale gestito
    // automaticamente
    for (int i = 2; i <= max_threads;
         i += 1) { // Scala esponenzialmente o linearmente
      threads_to_test.push_back(i);
    }
    if (threads_to_test.empty() || threads_to_test.back() != max_threads) {
      if (max_threads > 1)
        threads_to_test.push_back(
            max_threads); // Assicura che il massimo sia testato
    }
    if (max_threads == 1) {
      std::cout << "Warning: Only 1 hardware thread detected. Parallel "
                   "benchmarks might not show speedup."
                << std::endl;
    }

    // Chunk size da testare
    const std::vector<ull> chunks_to_test = {16,  32,  64,  96,
                                             128, 256, 512, 1024};

    // Definisci i workloads qui
    const std::vector<std::vector<Range>> workloads = {
        {{1, 50000}},   // Workload medio-piccolo bilanciato
        {{1, 1000000}}, // Workload grande bilanciato
        {{1, 100},
         {1000000, 1001000},
         {50000, 51000}}, // Sbilanciato (piccolo, grande, medio)
        []() {
          // Multipli range piccoli (generati programmaticamente)
          std::vector<Range> ranges;
          const ull range_size = 1000;
          const ull total_size = 100000;

          // Genera 100 range di 1000 numeri ciascuno
          for (ull start = 1; start < total_size; start += range_size) {
            ranges.push_back({start, start + range_size - 1});
          }
          return ranges;
        }(),
        {{9663, 9663},
         {77671, 77671},
         {626331, 626331},
         {837799, 837799}}, // Punti "difficili" (alti passi)
        []() {
          // Genera range intorno a diverse potenze di due
          std::vector<Range> ranges;
          const ull range_width = 1000; // Larghezza di ogni range

          // Crea range intorno alle potenze di 2 da 2^10 a 2^22
          for (int power = 10; power <= 22; power++) {
            ull center = 1ULL << power; // Potenza di 2
            ranges.push_back(
                {center - range_width / 2, center + range_width / 2});

            // Aggiungi anche un range poco prima e poco dopo per ogni potenza
            if (power > 10) { // Evita underflow per potenze piccole
              ranges.push_back(
                  {center - 3 * range_width, center - 2 * range_width});
              ranges.push_back(
                  {center + 2 * range_width, center + 3 * range_width});
            }
          }
          return ranges;
        }()};

    const std::vector<std::string> workload_descriptions = {
        "Medium Balanced",        "Large Balanced",
        "Unbalanced Mix",         "Multiple Small Balanced",
        "Known High-Step Points", "Multiple Power-of-2 Ranges"};

    // Parametri di misurazione
    const int samples = 10;               // Aumenta per risultati più stabili
    const int iterations_per_sample = 50; // Aumenta per ridurre varianza

    return run_benchmark_suite(threads_to_test, chunks_to_test, workloads,
                               workload_descriptions, samples,
                               iterations_per_sample);
  }

  // Flag non riconosciuto
  return false;
}

/**
 * @brief Main application entry point
 */
int main(int argc, char *argv[]) {
  // Gestisce --test-correctness o --benchmark
  if (is_test_or_benchmark_mode(argc, argv)) {
    bool success = handle_test_or_benchmark_mode(argc, argv);
    if (!success) {
      std::cerr << "Execution failed for flag '" << argv[1] << "'."
                << std::endl;
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }

  // Esecuzione normale: Parse arguments, execute, print results
  auto config_opt = parse_arguments(argc, argv);
  if (!config_opt) {
    return EXIT_FAILURE; // parse_arguments stampa l'errore
  }

  Config config = *config_opt;
  std::vector<RangeResult> results;
  Timer timer;

  bool success = execute_collatz_calculation(config, results);
  double elapsed_time_s = timer.elapsed_s();

  if (success) {
    print_results(results, config, elapsed_time_s);
    return EXIT_SUCCESS;
  } else {
    std::cerr << "Error during computation." << std::endl;
    return EXIT_FAILURE;
  }
}
