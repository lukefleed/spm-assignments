#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h" // For Timer
#include <algorithm>
#include <cmath>
#include <filesystem> // Requires C++17 for filesystem operations
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// === Configuration and Constants ===
namespace BenchmarkConfig {
const std::string RESULTS_DIR = "results/";
const std::string BENCHMARK_CSV_FILE =
    RESULTS_DIR + "performance_results_sencha.csv";
} // namespace BenchmarkConfig

// === Utility Functions (Internal Implementation Detail) ===
namespace TestUtils {

/**
 * @brief Creates a directory if it doesn't already exist.
 *        Handles potential errors during directory creation.
 * @param dir_path The path of the directory to ensure exists.
 * @throws std::runtime_error If directory creation fails.
 */
void ensure_directory_exists(const std::string &dir_path) {
  std::error_code ec;
  if (!std::filesystem::create_directories(dir_path, ec) && ec) {
    throw std::runtime_error("Failed to create directory: " + dir_path + " - " +
                             ec.message());
  }
}

/**
 * @brief Opens a CSV file for writing, ensuring the parent directory exists.
 *        Optionally writes a header row.
 * @param filename The full path to the CSV file.
 * @param header Optional header string to write as the first line.
 * @return std::ofstream An output file stream ready for writing.
 * @throws std::runtime_error If the directory cannot be created or the file
 * cannot be opened.
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
 * @brief Prints a formatted summary line for correctness test results.
 * @param testName Name of the test category being summarized.
 * @param total Total number of tests run in this category.
 * @param passed Number of tests that passed.
 */
void print_summary_line(const std::string &testName, int total, int passed) {
  std::cout << std::setw(25) << std::left << testName
            << " Total: " << std::setw(4) << total
            << " Passed: " << std::setw(4) << passed
            << " Failed: " << std::setw(4) << (total - passed) << std::endl;
}

/**
 * @brief Compares expected results (vector<ull>) from sequential execution
 *        with actual results (vector<RangeResult>) from a parallel scheduler.
 *
 * @param expected Vector of expected maximum step counts (ull).
 * @param actual Vector of RangeResult structs containing atomic max_steps.
 * @param test_id A string identifier for the specific test run (for error
 * messages).
 * @param verbose_error If true, print detailed error messages on mismatch.
 * @return true if all results match, false otherwise.
 */
bool compare_results(const std::vector<ull> &expected,
                     const std::vector<RangeResult> &actual,
                     const std::string &test_id, // Test identifier
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
    ull actual_value = actual[i].max_steps.load(std::memory_order_relaxed);
    if (actual_value != expected[i]) {
      if (verbose_error) {
        if (match) { // Print header only on the first detected error
          std::cerr << "\n  [" << test_id << "] Error: Mismatch at index " << i
                    << " (Range: " << actual[i].original_range.start << "-"
                    << actual[i].original_range.end << "). Expected "
                    << expected[i] << ", got " << actual_value << std::endl;
        }
      }
      match = false;
      // return false; // Optionally exit early on first mismatch
    }
  }
  return match;
}

} // namespace TestUtils

// === Core Benchmarking Structures ===

// Forward declaration
class ExperimentRunner;

struct Schedulable {
  using ExecutionFunc =
      std::function<bool(const Config &, std::vector<RangeResult> &)>;

  std::string name;
  std::string type_str;
  std::string variant_str;
  ExecutionFunc run_func;
  bool requires_threads;
  bool requires_chunk_size;

  SchedulingType type_enum = SchedulingType::SEQUENTIAL;
  StaticVariant static_variant_enum = StaticVariant::BLOCK;

  Schedulable(std::string n, std::string t_str, std::string v_str,
              ExecutionFunc func, bool needs_threads, bool needs_chunk)
      : name(std::move(n)), type_str(std::move(t_str)),
        variant_str(std::move(v_str)), run_func(std::move(func)),
        requires_threads(needs_threads), requires_chunk_size(needs_chunk) {
    if (type_str == "Sequential")
      type_enum = SchedulingType::SEQUENTIAL;
    else if (type_str == "Static")
      type_enum = SchedulingType::STATIC;
    else if (type_str == "Dynamic")
      type_enum = SchedulingType::DYNAMIC;

    if (type_enum == SchedulingType::STATIC) {
      if (variant_str == "Block")
        static_variant_enum = StaticVariant::BLOCK;
      else if (variant_str == "Cyclic")
        static_variant_enum = StaticVariant::CYCLIC;
      else if (variant_str == "Block-Cyclic")
        static_variant_enum = StaticVariant::BLOCK_CYCLIC;
    }
  }

  Config create_config(const std::vector<Range> &ranges, int threads,
                       ull chunk) const {
    Config cfg;
    cfg.ranges = ranges;
    cfg.scheduling = type_enum;
    cfg.static_variant =
        (type_enum == SchedulingType::STATIC)
            ? static_variant_enum
            : StaticVariant::BLOCK; // Use default for non-static
    cfg.num_threads = requires_threads ? threads : 1;

    if (requires_chunk_size) {
      cfg.chunk_size = (chunk > 0) ? chunk : 64; // Default to 64 if needed
                                                 // and chunk is 0/invalid
    } else {
      cfg.chunk_size = 0; // Explicitly 0 if not used
    }
    cfg.verbose = false;
    return cfg;
  }
};

class TimeMeasurer {
private:
  int samples;
  int iterations_per_sample;
  bool verbose;

public:
  TimeMeasurer(int s, int i, bool v = false)
      : samples(s), iterations_per_sample(i), verbose(v) {
    if (samples <= 0 || iterations_per_sample <= 0) {
      throw std::invalid_argument(
          "Samples and iterations must be positive integers.");
    }
  }

  double measure_median_time_ms(const Schedulable::ExecutionFunc &func_to_run,
                                const Config &config,
                                const std::string &measurement_id) {
    if (verbose) {
      std::cout << "    Measuring [" << measurement_id << "]..." << std::flush;
    }

    std::vector<double> valid_times_ms;
    valid_times_ms.reserve(samples * iterations_per_sample);
    std::vector<RangeResult> results_buffer;
    results_buffer.reserve(config.ranges.size());

    for (int s = 0; s < samples; ++s) {
      if (verbose)
        std::cout << " S" << (s + 1) << ":";
      int sample_successes = 0;
      for (int iter = 0; iter < iterations_per_sample; ++iter) {
        results_buffer.clear();
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
          std::cerr
              << "\n      Warning: Execution failed during measurement for ["
              << measurement_id << "], Sample " << s + 1 << ", Iter "
              << iter + 1 << std::endl;
        }
      }
      if (verbose && iterations_per_sample > 1) {
        std::cout << "(" << sample_successes << "/" << iterations_per_sample
                  << " ok) " << std::flush;
      }
    }

    if (valid_times_ms.empty()) {
      if (verbose)
        std::cout << " FAILED (No successful runs)" << std::endl;
      std::cerr << "\n      Error: Measurement failed for [" << measurement_id
                << "]. No successful iterations completed." << std::endl;
      return -1.0;
    }

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

class ExperimentRunner {
private:
  std::ofstream csv_file;
  TimeMeasurer measurer;
  std::string csv_filename;
  std::vector<std::string> workload_descriptions;

  const std::string CSV_HEADER =
      "WorkloadID,WorkloadDescription,SchedulerName,"
      "SchedulerType,StaticVariant,NumThreads,"
      "ChunkSize,ExecutionTimeMs,BaselineTimeMs,Speedup";

public:
  ExperimentRunner(std::string filename, int samples, int iterations,
                   const std::vector<std::string> &workload_descriptions)
      : measurer(samples, iterations,
                 /*verbose=*/false), // Set verbosity here if needed
        csv_filename(std::move(filename)),
        workload_descriptions(workload_descriptions) {
    csv_file = TestUtils::open_csv_file(csv_filename, CSV_HEADER);
  }

  bool run_suite(const Schedulable &baseline_schedulable,
                 const std::vector<Schedulable> &schedulables_to_test,
                 const std::vector<std::vector<Range>> &workloads,
                 const std::vector<int> &thread_counts,
                 const std::vector<ull> &chunk_sizes) {
    bool overall_success = true;

    if (workloads.size() != workload_descriptions.size()) {
      std::cerr << "Error: Mismatch between number of workloads ("
                << workloads.size() << ") and descriptions ("
                << workload_descriptions.size() << "). Aborting benchmark."
                << std::endl;
      return false;
    }

    for (size_t workload_idx = 0; workload_idx < workloads.size();
         ++workload_idx) {
      const auto current_workload = workloads[workload_idx];
      const auto current_description = workload_descriptions[workload_idx];
      std::cout << "\n--- Testing Workload " << workload_idx << ": "
                << current_description << " ---" << std::endl;
      std::cout << "  Ranges: ";
      for (const auto &r : current_workload) {
        std::cout << "[" << r.start << "-" << r.end << "] ";
      }
      std::cout << std::endl;

      // Establish Baseline
      std::cout << "  Establishing baseline (Sequential)..." << std::flush;
      Config baseline_config =
          baseline_schedulable.create_config(current_workload, 1, 0);
      std::string baseline_id =
          "Sequential Baseline W" + std::to_string(workload_idx);
      double baseline_time_ms = measurer.measure_median_time_ms(
          baseline_schedulable.run_func, baseline_config, baseline_id);

      if (baseline_time_ms <= 0) {
        std::cerr << " FAILED. Skipping workload." << std::endl;
        write_result(workload_idx, current_description, baseline_schedulable, 1,
                     0, -1.0, -1.0);
        overall_success = false;
        continue;
      }
      std::cout << " Done. Baseline Time: " << std::fixed
                << std::setprecision(4) << baseline_time_ms << " ms"
                << std::endl;
      write_result(workload_idx, current_description, baseline_schedulable, 1,
                   0, baseline_time_ms, baseline_time_ms);

      // Test other Schedulables
      for (const auto &sched : schedulables_to_test) {
        if (sched.name == baseline_schedulable.name)
          continue;

        const std::vector<int> &threads_to_test =
            sched.requires_threads ? thread_counts : std::vector<int>{1};
        const std::vector<ull> &chunks_to_test =
            sched.requires_chunk_size ? chunk_sizes : std::vector<ull>{0};

        for (int t : threads_to_test) {
          if (sched.requires_threads && t <= 1)
            continue;

          for (ull c : chunks_to_test) {
            // Skip invalid chunk sizes based on scheduler requirements
            if (sched.requires_chunk_size && c == 0)
              continue; // Dynamic needs > 0
            if (!sched.requires_chunk_size && c != 0)
              continue; // Static Block/Cyclic use 0 (N/A)

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
              // std::cout << std::endl; // End flush line from
              write_result(workload_idx, current_description, sched, t, c,
                           exec_time_ms, baseline_time_ms);
            } else {
              std::cout << " -> FAILED measurement/execution." << std::endl;
              write_result(workload_idx, current_description, sched, t, c, -1.0,
                           baseline_time_ms);
              overall_success = false;
            }
          }
        }
      }
    }

    finalize();
    return overall_success;
  }

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
    csv_file << "\"" << description << "\","; // Quote description
    csv_file << "\"" << sched.name << "\",";  // Quote scheduler name
    csv_file << sched.type_str << ",";
    csv_file << sched.variant_str << ",";
    csv_file << threads << ",";

    // Handle chunk size representation
    if (sched.requires_chunk_size && chunk_size > 0) {
      csv_file << chunk_size << ",";
    } else {
      csv_file << "N/A,";
    }

    // Handle time and speedup, indicating errors
    if (exec_time_ms > 0) {
      csv_file << std::fixed << std::setprecision(4) << exec_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << baseline_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << speedup;
    } else {
      csv_file << "Error," << std::fixed << std::setprecision(4)
               << baseline_time_ms << ",0.0";
    }
    csv_file << std::endl;
  }

  // finalize and destructor remain the same
  void finalize() {
    if (csv_file.is_open()) {
      csv_file.close();
      std::cout << "\nBenchmark results saved to " << csv_filename << std::endl;
    }
  }

  ~ExperimentRunner() { finalize(); }
};

namespace Schedulers {

bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  try {
    std::vector<ull> seq_results = run_sequential(cfg.ranges);
    res.clear();
    res.reserve(seq_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
      res.emplace_back(cfg.ranges[i]);
      res.back().max_steps.store(seq_results[i], std::memory_order_relaxed);
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

bool run_static_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  return run_static_scheduling(cfg, res);
}

// Wrapper for the dynamic WORK STEALING implementation
bool run_dynamic_work_stealing_wrapper(const Config &cfg,
                                       std::vector<RangeResult> &res) {
  // Directly forward the call
  return run_dynamic_work_stealing(cfg, res);
}

// Wrapper for the dynamic TASK QUEUE implementation
bool run_dynamic_task_queue_wrapper(const Config &cfg,
                                    std::vector<RangeResult> &res) {
  // Directly forward the call to the other dynamic function
  return run_dynamic_task_queue(cfg, res);
}

// --- MODIFY Schedulable Definitions ---

const Schedulable Sequential("Sequential", "Sequential", "N/A",
                             run_sequential_wrapper, false, false);
const Schedulable StaticBlock("Static Block", "Static", "Block",
                              run_static_wrapper, true, false);
const Schedulable StaticCyclic("Static Cyclic", "Static", "Cyclic",
                               run_static_wrapper, true, false);
const Schedulable StaticBlockCyclic("Static Block-Cyclic", "Static",
                                    "BlockCyclic", run_static_wrapper, true,
                                    true);

const Schedulable DynamicWorkStealing(
    "Dynamic WorkStealing",            // More descriptive name
    "Dynamic",                         // Type is still Dynamic
    "WorkStealing",                    // Variant identifies the implementation
    run_dynamic_work_stealing_wrapper, // Points to the correct wrapper
    true, true);

const Schedulable DynamicTaskQueue(
    "Dynamic TaskQueue",            // Descriptive name
    "Dynamic",                      // Type is also Dynamic
    "TaskQueue",                    // Variant identifies this implementation
    run_dynamic_task_queue_wrapper, // Points to the TaskQueue wrapper
    true, true);

const std::vector<Schedulable> AllSchedulers = {
    Sequential,        StaticBlock,      StaticCyclic,
    StaticBlockCyclic, DynamicTaskQueue, DynamicWorkStealing};

// Also include both in the parallel list for correctness tests
const std::vector<Schedulable> AllParallelSchedulers = {
    StaticBlock, StaticCyclic, StaticBlockCyclic, DynamicTaskQueue,
    DynamicWorkStealing};

} // namespace Schedulers

// === Correctness Testing Implementation ===

// CorrectnessTestCase struct remains the same
struct CorrectnessTestCase {
  std::string name;
  std::vector<Range> ranges;
  std::vector<int> thread_counts;
  std::vector<ull> chunk_sizes;
};

bool run_correctness_suite() {
  std::cout << "\n=== Running Correctness Suite ===" << std::endl;
  int total_cases = 0;
  int passed_cases = 0;

  // Define test cases (can remain the same)
  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {2, 4}, {1}},
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {2, 4, 8},
       {1, 5, 10}},
      {"Medium Range", {{1, 5000}}, {2, 8}, {64, 128}},
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {2, 4}, {16, 32}},
      {"Empty Range Input", {{50, 40}}, {2, 4}, {1}},
      {"Minimum Value", {{1, 1}}, {2, 4}, {1}},
      {"Large Chunk Size", {{1, 50}}, {2, 4}, {100, 200}},
      {"More Threads Than Items", {{1, 8}}, {16, 32}, {1, 2}},
      // Add a case specifically designed to challenge dynamic schedulers
      {"Highly Imbalanced",
       {{1, 100}, {100000, 100005}, {101, 200}},
       {4, 8},
       {1, 16}}};

  for (const auto &tc : test_cases) {
    total_cases++;
    bool case_passed_overall = true;
    std::cout << "\n[Test Case " << total_cases << ": " << tc.name << "]"
              << std::endl;
    std::cout << "  Ranges: ";
    for (const auto &r : tc.ranges)
      std::cout << "[" << r.start << "-" << r.end << "] ";
    std::cout << std::endl;

    // 1. Generate Baseline
    std::cout << "  Generating expected results (Sequential)..." << std::flush;
    std::vector<ull> expected_values;
    bool seq_success = false;
    try {
      expected_values = run_sequential(tc.ranges);
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
    if (!seq_success)
      continue;

    // 2. Test Parallel Schedulers
    int sub_test_count = 0;
    int sub_test_passed = 0;

    for (const auto &sched : Schedulers::AllParallelSchedulers) {
      // Determine parameters based on test case and scheduler requirements
      const std::vector<int> &threads_to_use = tc.thread_counts;

      std::vector<ull> chunks_to_use;
      if (sched.requires_chunk_size) {
        chunks_to_use = tc.chunk_sizes;
        chunks_to_use.erase(std::remove_if(chunks_to_use.begin(),
                                           chunks_to_use.end(),
                                           [](ull c) { return c == 0; }),
                            chunks_to_use.end());
        if (chunks_to_use.empty()) {
          chunks_to_use.push_back(1); // Default if none valid provided
        }
      } else {
        chunks_to_use = {0}; // Represents N/A
      }

      for (int t : threads_to_use) {
        if (t <= 1)
          continue; // Correctness test for parallel only

        for (ull c : chunks_to_use) {
          // Skip N/A chunk if scheduler requires one
          if (sched.requires_chunk_size && c == 0)
            continue;
          // Skip non-N/A chunk if scheduler doesn't require one
          if (!sched.requires_chunk_size && c != 0)
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
            result_match = TestUtils::compare_results(
                expected_values, actual_results, test_id, true);
          }

          if (run_success && result_match) {
            std::cout << " PASS" << std::endl;
            sub_test_passed++;
          } else {
            std::cout << " FAIL";
            if (!run_success)
              std::cout << " (Execution Error)." << error_msg;
            else
              std::cout << " (Result Mismatch).";
            std::cout << std::endl;
            case_passed_overall = false;
          }
        }
      }
    }

    std::cout << "  Case Summary: " << sub_test_passed << "/" << sub_test_count
              << " parallel configurations passed." << std::endl;

    if (case_passed_overall) {
      passed_cases++;
    }
  }

  std::cout << "\n=== Correctness Suite Summary ===" << std::endl;
  TestUtils::print_summary_line("Correctness Cases", total_cases, passed_cases);
  std::cout << "===================================" << std::endl;
  return (total_cases > 0 && total_cases == passed_cases);
}

// === Performance Benchmark Suite Implementation ===

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
    ExperimentRunner runner(BenchmarkConfig::BENCHMARK_CSV_FILE, samples,
                            iterations_per_sample, workload_descriptions);

    bool success =
        runner.run_suite(Schedulers::Sequential, Schedulers::AllSchedulers,
                         workloads, thread_counts, chunk_sizes);

    if (success) {
      std::cout << "\nBenchmark suite completed. Results are in the CSV file."
                << std::endl;
    } else {
      std::cout << "\nBenchmark suite completed, but some measurements or runs "
                   "encountered errors."
                << std::endl;
      std::cout << "Check console output and the CSV file ("
                << BenchmarkConfig::BENCHMARK_CSV_FILE
                << ") for details (e.g., 'Error' in time column)." << std::endl;
    }
    return success;

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
