#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem> // Requires C++17
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory> // For std::unique_ptr
#include <numeric>
#include <stdexcept> // For exceptions
#include <string>
#include <vector>

// === Configuration and Constants ===

namespace BenchmarkConfig {
const std::string RESULTS_DIR = "results/";
const std::string STATIC_COMPARISON_CSV =
    RESULTS_DIR + "static_performance_data.csv";
const std::string ALL_SCHEDULERS_CSV = RESULTS_DIR + "performance_data.csv";
const std::string WORKLOAD_SCALING_CSV =
    RESULTS_DIR + "workload_scaling_data.csv";

const int DEFAULT_SAMPLES = 5;
const int DEFAULT_ITERATIONS_PER_SAMPLE = 3;
const ull DEFAULT_FIXED_CHUNK_SIZE = 64;

// Default parameter ranges (can be overridden in main)
const std::vector<int> DEFAULT_THREAD_COUNTS = {1, 2, 4, 8, 16};
const std::vector<ull> DEFAULT_CHUNK_SIZES = {1, 16, 64, 128, 256};
const std::vector<Range> DEFAULT_WORKLOAD = {{1, 50000}}; // Example workload
} // namespace BenchmarkConfig

// === Utility Functions ===

namespace Utils {

/**
 * @brief Creates and ensures a directory exists.
 * @param dir_path The path to the directory.
 * @throws std::runtime_error if the directory cannot be created.
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
 * @param filename The full path to the CSV file.
 * @param header The CSV header row (optional).
 * @return An opened std::ofstream object.
 * @throws std::runtime_error if the directory cannot be created or the file
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
 * @brief Prints a summary line for test results.
 */
void print_summary_line(const std::string &testName, int total, int passed) {
  std::cout << std::setw(25) << std::left << testName
            << " Total: " << std::setw(4) << total
            << " Passed: " << std::setw(4) << passed
            << " Failed: " << std::setw(4) << (total - passed) << std::endl;
}

/**
 * @brief Compares expected sequential results with results from a scheduler
 * run.
 */
bool compare_results(const std::vector<ull> &expected,
                     const std::vector<RangeResult> &results,
                     const std::string &scheduler_id, // Use a unique ID
                     bool verbose_error = true) {
  if (results.size() != expected.size()) {
    if (verbose_error) {
      std::cerr << "  [" << scheduler_id
                << "] Error: Result count mismatch. Expected "
                << expected.size() << ", got " << results.size() << std::endl;
    }
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (results[i].max_steps.load() != expected[i]) {
      if (verbose_error) {
        std::cerr << "  [" << scheduler_id
                  << "] Error: Mismatch on range index " << i << ". Expected "
                  << expected[i] << ", got " << results[i].max_steps.load()
                  << std::endl;
      }
      return false; // Early exit on first mismatch
    }
  }
  return true;
}

} // namespace Utils

// === Core Benchmarking Structures ===

/**
 * @brief Represents the results of a single benchmark execution.
 */
struct BenchmarkResult {
  std::string scheduler_name;
  int thread_count = 0;
  ull chunk_size = 0; // 0 indicates N/A (e.g., for Sequential)
  double execution_time_ms = -1.0;
  double speedup = 0.0;
  bool success = false;
  int workload_idx = -1; // Optional: Index for workload scaling tests

  // Constructor
  BenchmarkResult(std::string name = "Unknown", int threads = 0, ull chunk = 0,
                  double time = -1.0, bool ok = false, double sp = 0.0,
                  int w_idx = -1)
      : scheduler_name(std::move(name)), thread_count(threads),
        chunk_size(chunk), execution_time_ms(time), speedup(sp), success(ok),
        workload_idx(w_idx) {}

  // Format result as a CSV row string
  std::string to_csv_row() const {
    std::stringstream ss;
    if (workload_idx >= 0) {
      ss << workload_idx << ",";
    }
    ss << scheduler_name << "," << thread_count << ",";
    if (chunk_size > 0) {
      ss << chunk_size;
    } else {
      ss << "N/A"; // Indicate chunk size not applicable
    }
    ss << ",";

    if (success) {
      ss << std::fixed << std::setprecision(4) << execution_time_ms << ","
         << std::fixed << std::setprecision(4) << speedup;
    } else {
      ss << "ERROR,0.0"; // Indicate failure
    }
    return ss.str();
  }
};

/**
 * @brief Abstract representation of a schedulable task/algorithm.
 */
struct Schedulable {
  using ExecutionFunc =
      std::function<bool(const Config &, std::vector<RangeResult> &)>;

  std::string name;       // User-friendly name (e.g., "Static Block")
  std::string id;         // Unique identifier (e.g., "static_block")
  ExecutionFunc run_func; // The function to execute the schedule
  bool requires_threads = false;
  bool requires_chunk_size = false;
  SchedulingType type = SchedulingType::SEQUENTIAL; // Default
  StaticVariant static_variant =
      StaticVariant::BLOCK; // Relevant if type is STATIC

  // Constructor for easier initialization
  Schedulable(std::string n, std::string i, ExecutionFunc func,
              bool needs_threads = false, bool needs_chunk = false,
              SchedulingType t = SchedulingType::SEQUENTIAL,
              StaticVariant sv = StaticVariant::BLOCK)
      : name(std::move(n)), id(std::move(i)), run_func(std::move(func)),
        requires_threads(needs_threads), requires_chunk_size(needs_chunk),
        type(t), static_variant(sv) {}

  // Creates a Config object tailored for this schedulable
  Config create_config(const Config &base_config, int threads,
                       ull chunk) const {
    Config cfg = base_config; // Copy base settings
    cfg.scheduling = type;
    if (type == SchedulingType::STATIC) {
      cfg.static_variant = static_variant;
    }
    cfg.num_threads = requires_threads ? threads : 1;
    cfg.chunk_size = requires_chunk_size ? chunk : 0;
    return cfg;
  }
};

/**
 * @brief Measures median execution time of a function with retries.
 */
class TimeMeasurer {
private:
  int samples;
  int iterations_per_sample;
  bool verbose;

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
   * @param scheduler_name Name for logging purposes.
   * @return Median time in milliseconds, or -1.0 on failure.
   */
  double measure_median_time_ms(const Schedulable::ExecutionFunc &func_to_run,
                                const Config &config,
                                const std::string &scheduler_name) {
    if (verbose) {
      std::cout << "  Measuring " << scheduler_name
                << " (T=" << config.num_threads << ", C="
                << (config.chunk_size > 0 ? std::to_string(config.chunk_size)
                                          : "N/A")
                << ")..." << std::endl;
    }

    std::vector<double> valid_times;
    valid_times.reserve(samples * iterations_per_sample);
    std::vector<RangeResult> results_buffer; // Reusable buffer

    for (int s = 0; s < samples; ++s) {
      if (verbose) {
        std::cout << "    Sample " << (s + 1) << "/" << samples << ": "
                  << std::flush;
      }
      int sample_successes = 0;
      for (int iter = 0; iter < iterations_per_sample; ++iter) {
        Timer timer;
        bool success = func_to_run(config, results_buffer);
        double duration_ms = timer.elapsed_ms();

        if (success) {
          valid_times.push_back(duration_ms);
          if (verbose)
            std::cout << "." << std::flush;
          sample_successes++;
        } else {
          if (verbose)
            std::cout << "X" << std::flush;
          // Optionally log the failure details here
        }
      }
      if (verbose)
        std::cout << " (" << sample_successes << "/" << iterations_per_sample
                  << " ok)" << std::endl;
    }

    if (valid_times.empty()) {
      if (verbose)
        std::cerr << "  Measurement failed: No successful iterations."
                  << std::endl;
      return -1.0; // Indicate failure
    }

    std::sort(valid_times.begin(), valid_times.end());
    size_t n = valid_times.size();
    double median = (n % 2 != 0)
                        ? valid_times[n / 2]
                        : (valid_times[n / 2 - 1] + valid_times[n / 2]) / 2.0;

    if (verbose) {
      std::cout << "  -> Median time (" << n << " valid runs): " << std::fixed
                << std::setprecision(4) << median << " ms" << std::endl;
    }

    return median;
  }
};

/**
 * @brief Manages running benchmark experiments and collecting results.
 */
class ExperimentRunner {
private:
  std::ofstream csv_file;
  Config base_config;
  std::string csv_filename;
  TimeMeasurer measurer;
  std::vector<BenchmarkResult> results;
  double baseline_time_ms = -1.0;
  std::string baseline_name = "Sequential"; // Default baseline

public:
  ExperimentRunner(std::string filename, int samples, int iterations,
                   const std::string &csv_header = "")
      : csv_filename(std::move(filename)),
        measurer(samples, iterations, false) // Internal measurer less verbose
  {
    csv_file = Utils::open_csv_file(csv_filename, csv_header);
    base_config.verbose = false; // Runner controls verbosity
  }

  // Set the workload (ranges) for the experiment
  void set_workload(const std::vector<Range> &ranges) {
    base_config.ranges = ranges;
  }

  // Establish the baseline time using a specific schedulable
  bool establish_baseline(const Schedulable &baseline_schedulable,
                          int workload_idx = -1) {
    std::cout << "\nEstablishing baseline using " << baseline_schedulable.name
              << "..." << std::flush;
    baseline_name = baseline_schedulable.name;

    // Baseline typically runs single-threaded, no chunking concept
    Config baseline_config =
        baseline_schedulable.create_config(base_config, 1, 0);

    baseline_time_ms = measurer.measure_median_time_ms(
        baseline_schedulable.run_func, baseline_config,
        baseline_schedulable.name);

    if (baseline_time_ms > 0) {
      std::cout << " Done. Baseline time: " << std::fixed
                << std::setprecision(4) << baseline_time_ms << " ms"
                << std::endl;
      BenchmarkResult baseline_result(baseline_name,
                                      baseline_config.num_threads,
                                      0, // Chunk N/A
                                      baseline_time_ms, true,
                                      1.0, // Speedup relative to itself is 1
                                      workload_idx);
      write_result(baseline_result);
      return true;
    } else {
      std::cerr << " FAILED." << std::endl;
      BenchmarkResult baseline_result(baseline_name,
                                      baseline_config.num_threads,
                                      0, // Chunk N/A
                                      -1.0, false, 0.0, workload_idx);
      write_result(baseline_result); // Write error to CSV
      return false;
    }
  }

  // Run a specific benchmark configuration
  BenchmarkResult run_benchmark(const Schedulable &schedulable, int threads,
                                ull chunk_size, int workload_idx = -1) {

    std::cout << "\nTesting " << schedulable.name << " (T=" << threads
              << ", C=" << (chunk_size > 0 ? std::to_string(chunk_size) : "N/A")
              << ")" << std::flush;

    Config run_config =
        schedulable.create_config(base_config, threads, chunk_size);

    double exec_time_ms = measurer.measure_median_time_ms(
        schedulable.run_func, run_config, schedulable.name);

    bool success = exec_time_ms > 0;
    double speedup = (success && baseline_time_ms > 0)
                         ? (baseline_time_ms / exec_time_ms)
                         : 0.0;

    std::cout << (success ? " -> Time: " + std::to_string(exec_time_ms) +
                                " ms, Speedup: " + std::to_string(speedup)
                          : " -> FAILED")
              << std::endl;

    return BenchmarkResult(schedulable.name, threads, chunk_size, exec_time_ms,
                           success, speedup, workload_idx);
  }

  // Run benchmarks for a list of schedulables over thread/chunk ranges
  void run_experiment(const std::vector<Schedulable> &schedulables_to_run,
                      const std::vector<int> &thread_counts,
                      const std::vector<ull> &chunk_sizes,
                      int workload_idx = -1) {

    if (baseline_time_ms <= 0) {
      std::cerr << "Warning: Baseline not established or failed. Speedup "
                   "calculation will be zero."
                << std::endl;
    }

    for (const auto &sched : schedulables_to_run) {
      const std::vector<int> &threads =
          sched.requires_threads ? thread_counts : std::vector<int>{1};
      const std::vector<ull> &chunks = sched.requires_chunk_size
                                           ? chunk_sizes
                                           : std::vector<ull>{0}; // 0 means N/A

      for (int t : threads) {
        for (ull c : chunks) {
          // Skip non-sensical combinations if needed (e.g., chunk size 0 for
          // dynamic)
          if (sched.type == SchedulingType::DYNAMIC && c == 0)
            continue;
          // Static block/cyclic don't strictly *need* chunk size, but often
          // tested with it Allow chunk 0 for static if needed, or enforce > 0:
          // if (sched.type == SchedulingType::STATIC && c == 0 &&
          // sched.static_variant != StaticVariant::BLOCK /*or CYCLIC*/)
          // continue;

          BenchmarkResult result = run_benchmark(sched, t, c, workload_idx);
          write_result(result);
        }
      }
    }
  }

  // Write a single result to the CSV
  void write_result(const BenchmarkResult &result) {
    if (csv_file.is_open()) {
      csv_file << result.to_csv_row() << std::endl;
    }
    results.push_back(result); // Also store internally if needed later
  }

  // Close the CSV file
  void finalize() {
    if (csv_file.is_open()) {
      csv_file.close();
      std::cout << "\nExperiment results saved to " << csv_filename
                << std::endl;
    }
  }

  ~ExperimentRunner() {
    finalize(); // Ensure file is closed on destruction
  }
};

// === Schedulable Definitions ===

namespace Schedulers {

// Wrapper for sequential implementation
bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  try {
    std::vector<ull> seq_results = run_sequential(cfg.ranges);
    res.clear();
    res.reserve(seq_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
      RangeResult rr(cfg.ranges[i]); // Create RangeResult with original range
      rr.max_steps.store(seq_results[i]);
      res.push_back(std::move(rr)); // Use move constructor
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Sequential execution failed: " << e.what() << std::endl;
    return false;
  }
}

// Wrapper for static scheduling (variant set in config)
bool run_static_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  return run_static_scheduling(cfg, res);
}

// Wrapper for dynamic scheduling
bool run_dynamic_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  return run_dynamic_task_queue(cfg, res);
}

// Define the standard set of schedulables
const Schedulable Sequential("Sequential", "seq", run_sequential_wrapper, false,
                             false, SchedulingType::SEQUENTIAL);
const Schedulable StaticBlock("Static Block", "static_block",
                              run_static_wrapper, true, true,
                              SchedulingType::STATIC, StaticVariant::BLOCK);
const Schedulable StaticCyclic("Static Cyclic", "static_cyclic",
                               run_static_wrapper, true, true,
                               SchedulingType::STATIC, StaticVariant::CYCLIC);
const Schedulable StaticBlockCyclic("Static Block-Cyclic",
                                    "static_block_cyclic", run_static_wrapper,
                                    true, true, SchedulingType::STATIC,
                                    StaticVariant::BLOCK_CYCLIC);
const Schedulable Dynamic("Dynamic", "dyn", run_dynamic_wrapper, true, true,
                          SchedulingType::DYNAMIC);

// Provide lists for common scenarios
const std::vector<Schedulable> AllStatic = {StaticBlock, StaticCyclic,
                                            StaticBlockCyclic};
const std::vector<Schedulable> AllParallel = {StaticBlock, StaticCyclic,
                                              StaticBlockCyclic, Dynamic};
const std::vector<Schedulable> StaticAndDynamic = {
    StaticBlockCyclic, Dynamic}; // Example: Compare best static vs dynamic

} // namespace Schedulers

// === Correctness Testing ===

/**
 * @brief Structure for correctness test case.
 */
struct CorrectnessTestCase {
  std::string name;
  std::vector<Range> ranges;
  std::vector<int> thread_counts;
  std::vector<ull> chunk_sizes;
};

/**
 * @brief Runs the full correctness test suite.
 */
bool run_correctness_suite() {
  std::cout << "\n=== Running Extended Correctness Suite ===" << std::endl;
  int total_cases = 0;
  int passed_cases = 0;

  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {1, 2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {1, 2, 4}, {1}},
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {1, 4, 8},
       {1, 10}},
      {"Larger Range", {{1, 10000}}, {1, 8, 16}, {64, 128}},
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {1, 4, 8}, {16}},
      {"Empty Range", {{50, 40}}, {1, 4}, {1}}, // Start > End
      {"Minimum Value", {{1, 1}}, {1, 4, 8}, {1, 16}},
      //    {"Boundary Case", {{4294967294, 4294967295}}, {1, 4}, {1, 64}}, //
      //    Requires ULL in Range? Check Collatz impl.
      {"Large Chunk Size", {{1, 100}}, {1, 2, 4}, {200, 500}}, // Chunk > range
      {"More Threads Than Work",
       {{1, 10}},
       {16, 32},
       {1, 4}}, // Threads > range size
      //    {"Zero Start", {{0, 10}}, {1, 4}, {1, 4}} // Check if Collatz
      //    handles 0 (usually starts from 1)
  };

  // Schedulers to test for correctness (all parallel ones)
  const std::vector<Schedulable> &schedulers_to_test = Schedulers::AllParallel;

  for (const auto &tc : test_cases) {
    total_cases++;
    bool case_passed = true;
    std::cout << "\n[Test Case " << total_cases << ": " << tc.name << "]"
              << std::endl;

    Config base_config;
    base_config.ranges = tc.ranges;
    base_config.verbose =
        false; // Correctness checks don't need timing verbosity

    std::cout << "  Generating expected results (Sequential)..." << std::flush;
    std::vector<RangeResult> expected_results_rr;
    bool seq_success =
        Schedulers::Sequential.run_func(base_config, expected_results_rr);

    if (!seq_success) {
      std::cerr << " FAILED (Sequential run error). Skipping case."
                << std::endl;
      case_passed = false;
    } else {
      std::cout << " Done." << std::endl;
      // Convert RangeResult back to simple ull vector for comparison ease
      std::vector<ull> expected_values;
      expected_values.reserve(expected_results_rr.size());
      for (const auto &rr : expected_results_rr) {
        expected_values.push_back(rr.max_steps.load());
      }

      int sub_test_count = 0;
      int sub_test_passed = 0;

      for (const auto &sched : schedulers_to_test) {
        const std::vector<int> &threads =
            sched.requires_threads ? tc.thread_counts : std::vector<int>{1};
        const std::vector<ull> &chunks =
            sched.requires_chunk_size ? tc.chunk_sizes : std::vector<ull>{0};

        for (int t : threads) {
          for (ull c : chunks) {
            if (sched.type == SchedulingType::DYNAMIC && c == 0)
              continue; // Skip invalid dynamic chunk

            sub_test_count++;
            std::string test_id = sched.name + " (T=" + std::to_string(t) +
                                  ", C=" + (c > 0 ? std::to_string(c) : "N/A") +
                                  ")";
            std::cout << "  Testing " << test_id << "..." << std::flush;

            Config run_config = sched.create_config(base_config, t, c);
            std::vector<RangeResult> actual_results;
            bool run_success = sched.run_func(run_config, actual_results);

            bool result_match = false;
            if (run_success) {
              result_match = Utils::compare_results(
                  expected_values, actual_results, test_id, true);
            }

            if (run_success && result_match) {
              std::cout << " PASS" << std::endl;
              sub_test_passed++;
            } else {
              std::cout << " FAIL" << (run_success ? "" : " (Execution Error)")
                        << std::endl;
              case_passed = false; // Mark the whole case as failed
            }
          }
        }
      }
      std::cout << "  Case Summary: " << sub_test_passed << "/"
                << sub_test_count << " configurations passed." << std::endl;
    }

    if (case_passed) {
      passed_cases++;
    }
  }

  std::cout << "\n=== Correctness Suite Summary ===" << std::endl;
  Utils::print_summary_line("Correctness Cases", total_cases, passed_cases);
  std::cout << "===================================" << std::endl;
  return (total_cases == passed_cases);
}

// === Performance Experiment Definitions ===

/**
 * @brief Runs static scheduler comparison tests.
 */
bool run_static_performance_comparison(const std::vector<int> &thread_counts,
                                       const std::vector<ull> &chunk_sizes,
                                       int samples, int iterations_per_sample,
                                       const std::vector<Range> &workload) {
  std::cout << "\n=== Running Static Scheduler Comparison ===" << std::endl;
  ExperimentRunner runner(BenchmarkConfig::STATIC_COMPARISON_CSV, samples,
                          iterations_per_sample,
                          "Scheduler,Threads,ChunkSize,TimeMs,Speedup");
  runner.set_workload(workload);

  if (!runner.establish_baseline(Schedulers::Sequential)) {
    return false;
  }

  runner.run_experiment(Schedulers::AllStatic, thread_counts, chunk_sizes);
  runner.finalize(); // Explicit finalize for clarity, though destructor handles
                     // it
  return true;
}

/**
 * @brief Runs performance tests comparing preferred static vs dynamic.
 */
bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload) {
  std::cout << "\n=== Running Performance Suite (Static vs Dynamic) ==="
            << std::endl;
  ExperimentRunner runner(BenchmarkConfig::ALL_SCHEDULERS_CSV, samples,
                          iterations_per_sample,
                          "Scheduler,Threads,ChunkSize,TimeMs,Speedup");
  runner.set_workload(workload);

  if (!runner.establish_baseline(Schedulers::Sequential)) {
    return false;
  }

  // Compare only BlockCyclic and Dynamic in this suite
  runner.run_experiment(Schedulers::StaticAndDynamic, thread_counts,
                        chunk_sizes);
  runner.finalize();
  return true;
}

/**
 * @brief Runs workload scaling tests across all parallel schedulers.
 */
bool run_workload_scaling_tests(
    const std::vector<int> &thread_counts,
    const std::vector<std::vector<Range>> &workloads, int samples,
    int iterations_per_sample) {
  std::cout << "\n=== Running Workload Scaling Tests ===" << std::endl;
  ExperimentRunner runner(
      BenchmarkConfig::WORKLOAD_SCALING_CSV, samples, iterations_per_sample,
      "WorkloadIdx,Scheduler,Threads,ChunkSize,TimeMs,Speedup");

  // Use a fixed chunk size for scaling tests for comparability
  const std::vector<ull> fixed_chunk_size = {
      BenchmarkConfig::DEFAULT_FIXED_CHUNK_SIZE};

  for (size_t i = 0; i < workloads.size(); ++i) {
    std::cout << "\n--- Workload Set " << i << " ---" << std::endl;
    runner.set_workload(workloads[i]);

    // Re-establish baseline for EACH workload
    if (!runner.establish_baseline(Schedulers::Sequential, i)) {
      std::cerr << "  Skipping performance runs for workload " << i
                << " due to baseline failure." << std::endl;
      continue; // Skip to next workload if baseline fails
    }

    // Run all parallel schedulers for this workload
    runner.run_experiment(Schedulers::AllParallel, thread_counts,
                          fixed_chunk_size, i);
  }

  runner.finalize();
  return true; // Indicate completion, success depends on individual runs
}
