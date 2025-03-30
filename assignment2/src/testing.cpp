#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h" // Per Timer

#include <algorithm>
#include <cmath>
#include <filesystem> // Requires C++17
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>    // Per mappare nomi a tipi
#include <memory> // For std::unique_ptr
#include <numeric>
#include <stdexcept> // For exceptions
#include <string>
#include <vector>

// === Configuration and Constants ===

namespace BenchmarkConfig {
const std::string RESULTS_DIR = "results/";
const std::string BENCHMARK_CSV_FILE =
    RESULTS_DIR + "performance_results.csv"; // Unico file CSV
// Default non più usati qui, passati da main
} // namespace BenchmarkConfig

// === Utility Functions (Namespace Interno) ===
namespace TestUtils {

/**
 * @brief Creates and ensures a directory exists.
 */
void ensure_directory_exists(const std::string &dir_path) {
  std::error_code ec;
  if (!std::filesystem::create_directories(dir_path, ec) && ec) {
    throw std::runtime_error("Failed to create directory: " + dir_path + " - " +
                             ec.message());
  }
}

/**
 * @brief Opens a CSV file for writing results, ensuring the directory exists.
 */
std::ofstream open_csv_file(const std::string &filename,
                            const std::string &header = "") {
  ensure_directory_exists(
      std::filesystem::path(filename).parent_path().string());
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open file " + filename +
                             " for writing.");
  }
  if (!header.empty()) {
    file << header << std::endl;
  }
  return file;
}

/**
 * @brief Prints a summary line for test results (usato da correctness).
 */
void print_summary_line(const std::string &testName, int total, int passed) {
  std::cout << std::setw(25) << std::left << testName
            << " Total: " << std::setw(4) << total
            << " Passed: " << std::setw(4) << passed
            << " Failed: " << std::setw(4) << (total - passed) << std::endl;
}

/**
 * @brief Compares expected sequential results (vector<ull>) with results from a
 * scheduler run (vector<RangeResult>).
 */
bool compare_results(const std::vector<ull> &expected,
                     const std::vector<RangeResult> &actual,
                     const std::string &test_id, // Identificatore del test
                     bool verbose_error = true) {
  if (actual.size() != expected.size()) {
    if (verbose_error) {
      std::cerr << "\n  [" << test_id
                << "] Error: Result count mismatch. Expected "
                << expected.size() << ", got " << actual.size() << std::endl;
    }
    return false;
  }
  bool match = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    // Confronta il valore atteso con il valore atomico caricato
    ull actual_value = actual[i].max_steps.load(std::memory_order_relaxed);
    if (actual_value != expected[i]) {
      if (verbose_error) {
        // Stampa solo il primo errore per non inondare l'output
        if (match) { // Stampa solo la prima volta che trova un errore
          std::cerr << "\n  [" << test_id << "] Error: Mismatch at index " << i
                    << " (Range: " << actual[i].original_range.start << "-"
                    << actual[i].original_range.end << "). Expected "
                    << expected[i] << ", got " << actual_value << std::endl;
        }
      }
      match = false;
      // Non fare 'return false' qui per poter eventualmente stampare altri
      // errori se si modifica la logica di verbose_error, ma per ora fermiamoci
      // al primo. return false; // Rimosso per permettere il controllo completo
      // opzionale
    }
  }
  return match;
}

} // namespace TestUtils

// === Core Benchmarking Structures ===

// Forward declaration
class ExperimentRunner;

/**
 * @brief Abstract representation of a schedulable task/algorithm.
 */
struct Schedulable {
  using ExecutionFunc =
      std::function<bool(const Config &, std::vector<RangeResult> &)>;

  std::string name;        // Nome leggibile (e.g., "Static Block-Cyclic")
  std::string type_str;    // "Static", "Dynamic", "Sequential"
  std::string variant_str; // "Block", "Cyclic", "BlockCyclic", "N/A"
  ExecutionFunc run_func;  // Funzione da eseguire
  bool requires_threads;
  bool requires_chunk_size;

  // Mappatura interna per il tipo e la variante (usata per creare Config)
  SchedulingType type_enum = SchedulingType::SEQUENTIAL;
  StaticVariant static_variant_enum = StaticVariant::BLOCK; // Default

  // Costruttore migliorato
  Schedulable(std::string n, std::string t_str, std::string v_str,
              ExecutionFunc func, bool needs_threads, bool needs_chunk)
      : name(std::move(n)), type_str(std::move(t_str)),
        variant_str(std::move(v_str)), run_func(std::move(func)),
        requires_threads(needs_threads), requires_chunk_size(needs_chunk) {
    // Determina enum basati sulle stringhe per coerenza
    if (type_str == "Sequential")
      type_enum = SchedulingType::SEQUENTIAL;
    else if (type_str == "Static")
      type_enum = SchedulingType::STATIC;
    else if (type_str == "Dynamic")
      type_enum = SchedulingType::DYNAMIC;
    // else rimarrà SEQUENTIAL (o gestisci errore)

    if (type_enum == SchedulingType::STATIC) {
      if (variant_str == "Block")
        static_variant_enum = StaticVariant::BLOCK;
      else if (variant_str == "Cyclic")
        static_variant_enum = StaticVariant::CYCLIC;
      else if (variant_str == "Block-Cyclic")
        static_variant_enum = StaticVariant::BLOCK_CYCLIC;
      // else rimarrà BLOCK (o gestisci errore)
    }
  }

  // Crea una Config specifica per questo Schedulable
  Config create_config(const std::vector<Range> &ranges, int threads,
                       ull chunk) const {
    Config cfg;
    cfg.ranges = ranges;
    cfg.scheduling = type_enum;
    cfg.static_variant = (type_enum == SchedulingType::STATIC)
                             ? static_variant_enum
                             : StaticVariant::BLOCK; // Default se non static
    cfg.num_threads = requires_threads ? threads : 1;
    // Assegna chunk_size solo se richiesto e > 0, altrimenti potrebbe essere 0
    // o un valore di default
    cfg.chunk_size = (requires_chunk_size && chunk > 0)
                         ? chunk
                         : 64; // Usa un default sensato se chunk non fornito o
                               // 0? o lascia 0? Decidiamo 64.
    if (!requires_chunk_size)
      cfg.chunk_size = 0; // Mette a 0 se non richiesto affatto
    cfg.verbose = false;  // Il runner controlla la verbosità
    return cfg;
  }
};

/**
 * @brief Measures median execution time of a function.
 */
class TimeMeasurer {
private:
  int samples;
  int iterations_per_sample;
  bool verbose; // Verbosity per la misurazione stessa

public:
  TimeMeasurer(int s, int i, bool v = false)
      : samples(s), iterations_per_sample(i), verbose(v) {
    if (samples <= 0 || iterations_per_sample <= 0) {
      throw std::invalid_argument("Samples and iterations must be positive.");
    }
  }

  /**
   * @brief Measures the median execution time.
   * @param func_to_run The function representing the scheduler execution.
   * @param config The configuration for the scheduler run.
   * @param measurement_id Identifier for verbose output.
   * @return Median time in milliseconds, or -1.0 on failure.
   */
  double measure_median_time_ms(const Schedulable::ExecutionFunc &func_to_run,
                                const Config &config,
                                const std::string &measurement_id) {
    if (verbose) {
      std::cout << "    Measuring [" << measurement_id << "]..." << std::flush;
    }

    std::vector<double> valid_times_ms;
    valid_times_ms.reserve(samples * iterations_per_sample);
    std::vector<RangeResult> results_buffer; // Buffer riutilizzabile

    for (int s = 0; s < samples; ++s) {
      if (verbose)
        std::cout << " S" << (s + 1) << ":";
      int sample_successes = 0;
      for (int iter = 0; iter < iterations_per_sample; ++iter) {
        results_buffer.clear(); // Pulisci buffer risultati per ogni run
        Timer timer;
        bool success = func_to_run(config, results_buffer);
        double duration_ms = timer.elapsed_ms();

        if (success) {
          valid_times_ms.push_back(duration_ms);
          if (verbose)
            std::cout << "." << std::flush;
          sample_successes++;
        } else {
          if (verbose)
            std::cout << "X" << std::flush;
          // Loggare l'errore qui potrebbe essere utile
          std::cerr
              << "\n      Warning: Execution failed during measurement for ["
              << measurement_id << "], Sample " << s + 1 << ", Iter "
              << iter + 1 << std::endl;
          // Potremmo decidere di invalidare l'intera misurazione se ci sono
          // troppi fallimenti
        }
        // Breve pausa per evitare surriscaldamento / throttling? Non necessario
        // di solito. std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (verbose &&
          iterations_per_sample > 1) { // Non stampare se solo 1 iterazione
        std::cout << "(" << sample_successes << "/" << iterations_per_sample
                  << " ok) " << std::flush;
      }
    }

    if (valid_times_ms.empty()) {
      if (verbose)
        std::cout << " FAILED (No successful runs)" << std::endl;
      std::cerr << "\n      Error: Measurement failed for [" << measurement_id
                << "]. No successful iterations completed." << std::endl;
      return -1.0; // Indica fallimento
    }

    // Calcola la mediana
    std::sort(valid_times_ms.begin(), valid_times_ms.end());
    size_t n = valid_times_ms.size();
    double median_ms =
        (n % 2 != 0)
            ? valid_times_ms[n / 2]
            : (valid_times_ms[n / 2 - 1] + valid_times_ms[n / 2]) / 2.0;

    if (verbose) {
      std::cout << " -> Median: " << std::fixed << std::setprecision(4)
                << median_ms << " ms (" << n << " valid runs)" << std::endl;
    }

    return median_ms;
  }
};

/**
 * @brief Manages running benchmark experiments and collecting results into a
 * CSV.
 */
class ExperimentRunner {
private:
  std::ofstream csv_file;
  TimeMeasurer measurer;
  std::string csv_filename;
  std::vector<std::string> workload_descriptions; // Store descriptions locally

  // Header CSV definito una volta
  const std::string CSV_HEADER =
      "WorkloadID,WorkloadDescription,SchedulerName,"
      "SchedulerType,StaticVariant,NumThreads,"
      "ChunkSize,ExecutionTimeMs,BaselineTimeMs,Speedup";

public:
  ExperimentRunner(std::string filename, int samples, int iterations,
                   const std::vector<std::string> &descriptions)
      : csv_filename(std::move(filename)),
        measurer(samples, iterations,
                 false), // Measurer interno non verboso di default
        workload_descriptions(descriptions) {
    csv_file = TestUtils::open_csv_file(csv_filename, CSV_HEADER);
  }

  /**
   * @brief Runs the entire benchmark suite for all workloads and
   * configurations.
   * @param baseline_schedulable The Schedulable representing sequential
   * execution.
   * @param schedulables_to_test Vector of all Schedulables to benchmark
   * (including baseline).
   * @param workloads Vector of workloads (each a vector<Range>).
   * @param thread_counts Vector of thread counts to test.
   * @param chunk_sizes Vector of chunk sizes to test.
   */
  bool run_suite(const Schedulable &baseline_schedulable,
                 const std::vector<Schedulable> &schedulables_to_test,
                 const std::vector<std::vector<Range>> &workloads,
                 const std::vector<int> &thread_counts,
                 const std::vector<ull> &chunk_sizes) {
    bool overall_success = true;

    if (workloads.size() != workload_descriptions.size()) {
      std::cerr << "Error: Mismatch between number of workloads ("
                << workloads.size() << ") and descriptions ("
                << workload_descriptions.size() << ")." << std::endl;
      return false;
    }

    for (size_t workload_idx = 0; workload_idx < workloads.size();
         ++workload_idx) {
      const auto &current_workload = workloads[workload_idx];
      const auto &current_description = workload_descriptions[workload_idx];

      std::cout << "\n--- Testing Workload " << workload_idx << ": "
                << current_description << " ---" << std::endl;
      std::cout << "  Ranges: ";
      for (const auto &r : current_workload) {
        std::cout << "[" << r.start << "-" << r.end << "] ";
      }
      std::cout << std::endl;

      // 1. Stabilisci il baseline time PER QUESTO WORKLOAD
      std::cout << "  Establishing baseline (Sequential)..." << std::flush;
      Config baseline_config =
          baseline_schedulable.create_config(current_workload, 1, 0);
      double baseline_time_ms = measurer.measure_median_time_ms(
          baseline_schedulable.run_func, baseline_config,
          "Sequential Baseline W" + std::to_string(workload_idx));

      if (baseline_time_ms <= 0) {
        std::cerr << " FAILED. Cannot proceed with this workload." << std::endl;
        // Scrivi un record di errore per il baseline? Opzionale.
        write_result(workload_idx, current_description, baseline_schedulable, 1,
                     0, -1.0, -1.0); // Indica errore
        overall_success = false;
        continue; // Salta al prossimo workload
      }
      std::cout << " Done. Baseline Time: " << std::fixed
                << std::setprecision(4) << baseline_time_ms << " ms"
                << std::endl;

      // Scrivi il risultato del baseline nel CSV
      write_result(workload_idx, current_description, baseline_schedulable, 1,
                   0, baseline_time_ms, baseline_time_ms); // Speedup è 1

      // 2. Testa tutte le altre configurazioni per questo workload
      for (const auto &sched : schedulables_to_test) {
        // Salta il baseline, l'abbiamo già fatto
        if (sched.name == baseline_schedulable.name)
          continue;

        // Determina i thread e chunk size da usare per questo scheduler
        const std::vector<int> &threads =
            sched.requires_threads
                ? thread_counts
                : std::vector<int>{1}; // Dovrebbe essere sempre >1 qui
        const std::vector<ull> &chunks =
            sched.requires_chunk_size
                ? chunk_sizes
                : std::vector<ull>{0}; // 0 significa N/A o default

        for (int t : threads) {
          // Assicurati che t sia > 1 se lo scheduler richiede threads
          if (sched.requires_threads && t <= 1)
            continue; // Salta t=1 per scheduler paralleli

          for (ull c : chunks) {
            // Salta chunk 0 per Dynamic che lo richiede > 0
            if (sched.type_enum == SchedulingType::DYNAMIC && c == 0)
              continue;
            // Se lo scheduler non richiede chunk, esegui solo una volta con c=0
            // (N/A)
            if (!sched.requires_chunk_size && c != 0)
              continue;

            std::string run_id = sched.name + " (T=" + std::to_string(t) +
                                 ", C=" + (c > 0 ? std::to_string(c) : "N/A") +
                                 ", W" + std::to_string(workload_idx) + ")";
            std::cout << "  Testing " << sched.name << " (T=" << t
                      << ", C=" << (c > 0 ? std::to_string(c) : "N/A") << ")..."
                      << std::flush;

            Config run_config = sched.create_config(current_workload, t, c);
            double exec_time_ms = measurer.measure_median_time_ms(
                sched.run_func, run_config, run_id);

            if (exec_time_ms > 0) {
              std::cout << " -> Time: " << std::fixed << std::setprecision(4)
                        << exec_time_ms << " ms" << std::endl;
              write_result(workload_idx, current_description, sched, t, c,
                           exec_time_ms, baseline_time_ms);
            } else {
              std::cout << " -> FAILED execution." << std::endl;
              // Scrivi un risultato di errore nel CSV
              write_result(workload_idx, current_description, sched, t, c, -1.0,
                           baseline_time_ms); // Tempo -1 indica errore
              overall_success = false; // Segna che qualcosa è andato storto
            }
          } // end chunk loop
        } // end thread loop
      } // end scheduler loop
    } // end workload loop

    finalize();
    return overall_success;
  }

  // Scrive una singola riga di risultato nel CSV
  void write_result(size_t workload_idx, const std::string &description,
                    const Schedulable &sched, int threads, ull chunk_size,
                    double exec_time_ms, double baseline_time_ms) {
    if (!csv_file.is_open())
      return;

    double speedup = 0.0;
    if (exec_time_ms > 0 && baseline_time_ms > 0) {
      speedup = baseline_time_ms / exec_time_ms;
    }

    csv_file << workload_idx << ",";
    csv_file << "\"" << description
             << "\","; // Racchiudi la descrizione tra virgolette per gestire
                       // eventuali virgole
    csv_file << "\"" << sched.name << "\",";
    csv_file << sched.type_str << ",";
    csv_file << sched.variant_str << ",";
    csv_file << threads << ",";
    if (sched.requires_chunk_size && chunk_size > 0) {
      csv_file << chunk_size << ",";
    } else {
      csv_file << "N/A,"; // O 0, scegli la rappresentazione preferita
    }

    // Stampa tempi e speedup
    if (exec_time_ms > 0) {
      csv_file << std::fixed << std::setprecision(4) << exec_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << baseline_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << speedup;
    } else {
      csv_file << "Error," << std::fixed << std::setprecision(4)
               << baseline_time_ms << ",0.0"; // Indica errore nell'esecuzione
    }
    csv_file << std::endl;
  }

  // Chiude il file CSV
  void finalize() {
    if (csv_file.is_open()) {
      csv_file.close();
      std::cout << "\nBenchmark results saved to " << csv_filename << std::endl;
    }
  }

  ~ExperimentRunner() {
    finalize(); // Assicura che il file venga chiuso
  }
};

// === Schedulable Definitions (Namespace Interno) ===

namespace Schedulers {

// Wrapper per sequential implementation
// Serve ancora perché Schedulable richiede una funzione con firma (Config,
// vector<RangeResult>&)
bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  try {
    std::vector<ull> seq_results = run_sequential(cfg.ranges);
    res.clear();
    res.reserve(seq_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
      res.emplace_back(
          cfg.ranges[i]); // Crea RangeResult con il range originale
      res.back().max_steps.store(
          seq_results[i], std::memory_order_relaxed); // Usa store rilassato
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "\nError in run_sequential_wrapper: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "\nUnknown error in run_sequential_wrapper." << std::endl;
    return false;
  }
}

// Wrapper per static scheduling (la variante è nella Config)
bool run_static_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  return run_static_scheduling(cfg, res);
}

// Wrapper per dynamic scheduling
bool run_dynamic_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  return run_dynamic_task_queue(cfg, res);
}

// Definisci l'insieme standard di scheduler da testare
const Schedulable Sequential("Sequential", "Sequential", "N/A",
                             run_sequential_wrapper, false, false);
const Schedulable StaticBlock(
    "Static Block", "Static", "Block", run_static_wrapper, true,
    true); // Richiede chunk per BlockCyclic, quindi lo mettiamo true
const Schedulable StaticCyclic("Static Cyclic", "Static", "Cyclic",
                               run_static_wrapper, true,
                               false); // Cyclic puro non usa chunk size
const Schedulable StaticBlockCyclic("Static Block-Cyclic", "Static",
                                    "BlockCyclic", // Cambiato nome variant
                                    run_static_wrapper, true,
                                    true); // Richiede chunk size
const Schedulable Dynamic("Dynamic", "Dynamic", "N/A", run_dynamic_wrapper,
                          true, true); // Richiede chunk size

// Lista di tutti gli scheduler per il benchmark
const std::vector<Schedulable> AllSchedulers = {
    Sequential, // Includi Sequential così i suoi tempi sono nel CSV per
                // confronto diretto
    StaticBlock, StaticCyclic, StaticBlockCyclic, Dynamic};

// Lista degli scheduler paralleli (per i test di correttezza)
const std::vector<Schedulable> AllParallelSchedulers = {
    StaticBlock, StaticCyclic, StaticBlockCyclic, Dynamic};

} // namespace Schedulers

// === Correctness Testing Implementation ===

/**
 * @brief Structure for correctness test case.
 */
struct CorrectnessTestCase {
  std::string name;
  std::vector<Range> ranges;
  std::vector<int> thread_counts; // Threads da testare (oltre a 1)
  std::vector<ull> chunk_sizes;   // Chunks da testare
};

/**
 * @brief Runs the full correctness test suite.
 */
bool run_correctness_suite() {
  std::cout << "\n=== Running Correctness Suite ===" << std::endl;
  int total_cases = 0;
  int passed_cases = 0;

  // Casi di test ampliati e rivisti
  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {2, 4}, {1}}, // Valore noto
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {2, 4, 8},
       {1, 5, 10}},
      {"Medium Range", {{1, 5000}}, {2, 8}, {64, 128}},
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {2, 4}, {16, 32}},
      {"Empty Range Input",
       {{50, 40}},
       {2, 4},
       {1}}, // Start > End (deve produrre 0 passi o gestire correttamente)
      {"Minimum Value", {{1, 1}}, {2, 4}, {1}},
      {"Large Chunk Size", {{1, 50}}, {2, 4}, {100, 200}}, // Chunk > range
      {"More Threads Than Work Items",
       {{1, 8}},
       {16, 32},
       {1, 2}}, // Più thread che numeri
      {"Range with Zero (if supported)",
       {{0, 10}},
       {2, 4},
       {1, 4}} // Verifica se Collatz(0) è gestito (di solito non lo è)
               // Se 0 non è valido, il test deve aspettarsi un fallimento o un
               // risultato specifico
  };

  for (const auto &tc : test_cases) {
    total_cases++;
    bool case_passed_overall = true;
    std::cout << "\n[Test Case " << total_cases << ": " << tc.name << "]"
              << std::endl;
    std::cout << "  Ranges: ";
    for (const auto &r : tc.ranges) {
      std::cout << "[" << r.start << "-" << r.end << "] ";
    }
    std::cout << std::endl;

    // 1. Ottieni i risultati attesi dall'esecuzione sequenziale
    std::cout << "  Generating expected results (Sequential)..." << std::flush;
    Config base_config; // Crea una config base solo con i range
    base_config.ranges = tc.ranges;
    base_config.verbose = false;

    std::vector<RangeResult> expected_results_rr;
    std::vector<ull> expected_values;
    // Usa direttamente run_sequential, non il wrapper, per chiarezza qui
    bool seq_success = false;
    try {
      std::vector<ull> seq_raw_results = run_sequential(tc.ranges);
      // Converti in expected_values
      expected_values = seq_raw_results;
      seq_success = true;
      std::cout << " Done (" << expected_values.size() << " results)."
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << " FAILED (Sequential execution error: " << e.what()
                << "). Skipping case." << std::endl;
      case_passed_overall = false;
    } catch (...) {
      std::cerr
          << " FAILED (Unknown sequential execution error). Skipping case."
          << std::endl;
      case_passed_overall = false;
    }

    if (!seq_success) {
      continue; // Vai al prossimo test case
    }

    // 2. Esegui ogni scheduler parallelo con le combinazioni di thread/chunk
    int sub_test_count = 0;
    int sub_test_passed = 0;

    for (const auto &sched : Schedulers::AllParallelSchedulers) {

      // Usa i thread_counts dal test case
      const std::vector<int> &threads_to_use = tc.thread_counts;
      // Usa i chunk_sizes dal test case, aggiungendo 0 se non richiesto dallo
      // scheduler
      std::vector<ull> chunks_to_use;
      if (sched.requires_chunk_size) {
        chunks_to_use = tc.chunk_sizes;
        // Rimuovi 0 se presente, dato che lo scheduler lo richiede > 0
        chunks_to_use.erase(
            std::remove(chunks_to_use.begin(), chunks_to_use.end(), 0),
            chunks_to_use.end());
        if (chunks_to_use.empty())
          chunks_to_use.push_back(
              1); // Assicura almeno un chunk size valido se richiesto
      } else {
        chunks_to_use = {0}; // 0 significa N/A per questo scheduler
      }

      for (int t : threads_to_use) {
        if (t <= 1)
          continue; // Testiamo solo > 1 thread per paralleli

        for (ull c : chunks_to_use) {
          // Salta chunk 0 per Dynamic
          if (sched.type_enum == SchedulingType::DYNAMIC && c == 0)
            continue;

          sub_test_count++;
          std::string test_id = sched.name + " (T=" + std::to_string(t) +
                                ", C=" + (c > 0 ? std::to_string(c) : "N/A") +
                                ")";
          std::cout << "    Testing " << test_id << "..." << std::flush;

          Config run_config = sched.create_config(tc.ranges, t, c);
          std::vector<RangeResult> actual_results;
          bool run_success = false;
          std::string error_msg;

          try {
            run_success = sched.run_func(run_config, actual_results);
          } catch (const std::exception &e) {
            run_success = false;
            error_msg = " Exception: " + std::string(e.what());
          } catch (...) {
            run_success = false;
            error_msg = " Unknown exception";
          }

          bool result_match = false;
          if (run_success) {
            // Confronta i risultati con quelli attesi
            result_match = TestUtils::compare_results(
                expected_values, actual_results, test_id, true);
          }

          if (run_success && result_match) {
            std::cout << " PASS" << std::endl;
            sub_test_passed++;
          } else {
            std::cout << " FAIL";
            if (!run_success) {
              std::cout << " (Execution Error)." << error_msg;
            } else if (!result_match) {
              std::cout << " (Result Mismatch).";
              // compare_results ha già stampato il dettaglio dell'errore
            }
            std::cout << std::endl;
            case_passed_overall = false; // Segna l'intero caso come fallito
          }
        } // end chunk loop
      } // end thread loop
    } // end scheduler loop

    std::cout << "  Case Summary: " << sub_test_passed << "/" << sub_test_count
              << " parallel configurations passed." << std::endl;

    if (case_passed_overall) {
      passed_cases++;
    }
  } // end test case loop

  std::cout << "\n=== Correctness Suite Summary ===" << std::endl;
  TestUtils::print_summary_line("Correctness Cases", total_cases, passed_cases);
  std::cout << "===================================" << std::endl;
  return (total_cases == passed_cases &&
          total_cases > 0); // Ritorna true solo se tutti i casi passano
}

// === Performance Benchmark Suite Implementation ===

/**
 * @brief Runs the main performance benchmark suite.
 */
bool run_benchmark_suite(const std::vector<int> &thread_counts,
                         const std::vector<ull> &chunk_sizes,
                         const std::vector<std::vector<Range>> &workloads,
                         const std::vector<std::string> &workload_descriptions,
                         int samples, int iterations_per_sample) {
  std::cout << "\n=== Running Performance Benchmark Suite ===" << std::endl;
  std::cout << "Saving results to: " << BenchmarkConfig::BENCHMARK_CSV_FILE
            << std::endl;
  std::cout << "Parameters: Samples=" << samples
            << ", Iterations/Sample=" << iterations_per_sample << std::endl;
  std::cout << "Threads to test: ";
  for (int t : thread_counts)
    std::cout << t << " ";
  std::cout << std::endl;
  std::cout << "Chunk sizes to test: ";
  for (ull c : chunk_sizes)
    std::cout << c << " ";
  std::cout << std::endl;

  try {
    // Crea l'ExperimentRunner
    ExperimentRunner runner(BenchmarkConfig::BENCHMARK_CSV_FILE, samples,
                            iterations_per_sample, workload_descriptions);

    // Esegui la suite completa
    bool success =
        runner.run_suite(Schedulers::Sequential,    // Il baseline scheduler
                         Schedulers::AllSchedulers, // Tutti da testare
                         workloads, thread_counts, chunk_sizes);

    if (success) {
      std::cout << "\nBenchmark suite completed successfully." << std::endl;
    } else {
      std::cout << "\nBenchmark suite completed with some errors or failures."
                << std::endl;
      std::cout << "Check the output and the CSV file ("
                << BenchmarkConfig::BENCHMARK_CSV_FILE << ") for details."
                << std::endl;
    }
    return success; // Ritorna lo stato generale

  } catch (const std::exception &e) {
    std::cerr << "\nFATAL ERROR during benchmark execution: " << e.what()
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "\nFATAL UNKNOWN ERROR during benchmark execution."
              << std::endl;
    return false;
  }
}
