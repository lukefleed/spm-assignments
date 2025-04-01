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
#include <map>    // Used for mapping names to types, although less critical now
#include <memory> // For std::unique_ptr (though not explicitly used in this snippet, good include)
#include <numeric>
#include <stdexcept> // For standard exceptions like std::runtime_error
#include <string>
#include <vector>

// === Configuration and Constants ===

/**
 * @brief Global configuration settings for the benchmark execution.
 */
namespace BenchmarkConfig {
/** @brief Directory where benchmark result files will be saved. */
const std::string RESULTS_DIR = "results/";
/** @brief The single CSV file consolidating all performance benchmark results.
 */
const std::string BENCHMARK_CSV_FILE =
    RESULTS_DIR + "performance_results_sencha_new_dynamic.csv";
// Note: Default parameters like thread counts or chunk sizes are now expected
// to be passed via main() rather than being hardcoded here.
} // namespace BenchmarkConfig

// === Utility Functions (Internal Implementation Detail) ===

/**
 * @brief Internal utility functions for testing and file handling.
 */
namespace TestUtils {

/**
 * @brief Creates a directory if it doesn't already exist.
 *        Handles potential errors during directory creation.
 * @param dir_path The path of the directory to ensure exists.
 * @throws std::runtime_error If directory creation fails.
 */
void ensure_directory_exists(const std::string &dir_path) {
  std::error_code ec;
  // Attempt to create the directory and its parents if necessary.
  // Check if creation failed *and* an error code was set (to distinguish
  // from the case where the directory already existed).
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
  // Ensure the directory containing the target file exists before opening.
  ensure_directory_exists(
      std::filesystem::path(filename).parent_path().string());

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open file " + filename +
                             " for writing.");
  }
  // Write the header if provided.
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
 * @note Uses std::memory_order_relaxed for loading the atomic 'max_steps'.
 *       This is sufficient and performant because the comparison happens
 * *after* all worker threads have finished and their writes are guaranteed to
 * be visible (due to thread join synchronization in the calling scheduler). We
 * only need the final value, not strict ordering during execution.
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
    // Load the atomic result using relaxed memory order (see function note).
    ull actual_value = actual[i].max_steps.load(std::memory_order_relaxed);
    if (actual_value != expected[i]) {
      if (verbose_error) {
        // Only print the first mismatch detail per comparison to avoid flooding
        // output.
        if (match) { // Print header only on the first detected error
          std::cerr << "\n  [" << test_id << "] Error: Mismatch at index " << i
                    << " (Range: " << actual[i].original_range.start << "-"
                    << actual[i].original_range.end << "). Expected "
                    << expected[i] << ", got " << actual_value << std::endl;
        }
      }
      match = false;
      // Optimization: Could return false immediately here, but completing the
      // loop might be useful if verbose_error logic were changed to show all
      // errors. For now, finding one mismatch is sufficient to declare failure.
      // return false; // Keep commented to allow potential future changes.
    }
  }
  return match;
}

} // namespace TestUtils

// === Core Benchmarking Structures ===

// Forward declaration
class ExperimentRunner;

/**
 * @brief Abstract representation of a schedulable unit of work
 * (algorithm/scheduler). Provides a common interface for different scheduling
 * strategies.
 */
struct Schedulable {
  /**
   * @brief Type alias for the function signature required to execute a
   * schedule.
   * @param config The configuration for this specific run.
   * @param results Output vector to store the results (RangeResult).
   * @return true on successful execution, false on failure.
   */
  using ExecutionFunc =
      std::function<bool(const Config &, std::vector<RangeResult> &)>;

  std::string name;     // User-friendly name (e.g., "Static Block-Cyclic")
  std::string type_str; // Category: "Static", "Dynamic", "Sequential"
  std::string
      variant_str; // Specific variant: "Block", "Cyclic", "BlockCyclic", "N/A"
  ExecutionFunc run_func;   // The function to call for execution.
  bool requires_threads;    // True if the scheduler uses >1 thread.
  bool requires_chunk_size; // True if the scheduler uses a chunk_size parameter
                            // > 0.

  // Internal enum representations for type and variant, used for creating
  // Config.
  SchedulingType type_enum = SchedulingType::SEQUENTIAL;
  StaticVariant static_variant_enum =
      StaticVariant::BLOCK; // Default if not static

  /**
   * @brief Constructs a Schedulable object.
   * @param n User-friendly name.
   * @param t_str Type string ("Sequential", "Static", "Dynamic").
   * @param v_str Variant string ("Block", "Cyclic", "BlockCyclic", "N/A").
   * @param func The execution function wrapper.
   * @param needs_threads Whether the scheduler is inherently parallel.
   * @param needs_chunk Whether the scheduler requires a chunk size parameter.
   */
  Schedulable(std::string n, std::string t_str, std::string v_str,
              ExecutionFunc func, bool needs_threads, bool needs_chunk)
      : name(std::move(n)), type_str(std::move(t_str)),
        variant_str(std::move(v_str)), run_func(std::move(func)),
        requires_threads(needs_threads), requires_chunk_size(needs_chunk) {
    // Set internal enums based on provided strings for consistency.
    if (type_str == "Sequential")
      type_enum = SchedulingType::SEQUENTIAL;
    else if (type_str == "Static")
      type_enum = SchedulingType::STATIC;
    else if (type_str == "Dynamic")
      type_enum = SchedulingType::DYNAMIC;
    // else: It remains SEQUENTIAL, or an error could be thrown if invalid type.

    if (type_enum == SchedulingType::STATIC) {
      if (variant_str == "Block")
        static_variant_enum = StaticVariant::BLOCK;
      else if (variant_str == "Cyclic")
        static_variant_enum = StaticVariant::CYCLIC;
      else if (variant_str ==
               "Block-Cyclic") // Ensure hyphenated name consistency
        static_variant_enum = StaticVariant::BLOCK_CYCLIC;
      // else: It remains BLOCK, or handle error.
    }
  }

  /**
   * @brief Creates a Config object tailored for this Schedulable.
   * @param ranges The input ranges for the calculation.
   * @param threads The number of threads to configure (used if requires_threads
   * is true).
   * @param chunk The chunk size to configure (used if requires_chunk_size is
   * true).
   * @return A Config struct populated for this Schedulable type.
   * @note Sets chunk_size based on requirements. A default of 64 is used if
   *       chunk > 0 is needed but not provided (or is 0). Chunk size 0 is
   *       explicitly set if the scheduler doesn't use it (e.g., Static Cyclic).
   *       The default 64 is arbitrary but common (related to cache line sizes).
   */
  Config create_config(const std::vector<Range> &ranges, int threads,
                       ull chunk) const {
    Config cfg;
    cfg.ranges = ranges;
    cfg.scheduling = type_enum;
    cfg.static_variant =
        (type_enum == SchedulingType::STATIC)
            ? static_variant_enum
            : StaticVariant::BLOCK; // Default if not static type
    cfg.num_threads = requires_threads ? threads : 1;

    // Assign chunk_size carefully based on scheduler requirements.
    if (requires_chunk_size) {
      // Use provided chunk if > 0, otherwise use a default (e.g., 64).
      // A chunk size must be > 0 for schedulers that require it (like Dynamic).
      cfg.chunk_size = (chunk > 0) ? chunk : 64;
    } else {
      // Explicitly set chunk_size to 0 if the scheduler doesn't use it.
      cfg.chunk_size = 0;
    }
    cfg.verbose =
        false; // Verbosity is controlled by the ExperimentRunner/caller.
    return cfg;
  }
};

/**
 * @brief Measures the execution time of a function, focusing on the median.
 *        Using the median provides robustness against outliers from system
 * noise.
 */
class TimeMeasurer {
private:
  int samples;               // Number of independent samples to collect.
  int iterations_per_sample; // Number of runs within each sample.
  bool verbose;              // Enable verbose output during measurement.

public:
  /**
   * @brief Constructs a TimeMeasurer.
   * @param s Number of samples.
   * @param i Number of iterations per sample.
   * @param v Enable verbose output during timing.
   * @throws std::invalid_argument If s or i are not positive.
   */
  TimeMeasurer(int s, int i, bool v = false)
      : samples(s), iterations_per_sample(i), verbose(v) {
    if (samples <= 0 || iterations_per_sample <= 0) {
      throw std::invalid_argument(
          "Samples and iterations must be positive integers.");
    }
  }

  /**
   * @brief Measures the median execution time of a Schedulable's run function.
   *
   * @param func_to_run The Schedulable::ExecutionFunc to time.
   * @param config The configuration to pass to the execution function.
   * @param measurement_id A string identifier for logging purposes.
   * @return Median execution time in milliseconds, or -1.0 if no successful
   *         runs occurred.
   * @note Runs the function multiple times (`samples * iterations_per_sample`)
   *       and calculates the median of successful runs. This approach helps
   *       mitigate the impact of outliers (e.g., cold starts, OS interference).
   */
  double measure_median_time_ms(const Schedulable::ExecutionFunc &func_to_run,
                                const Config &config,
                                const std::string &measurement_id) {
    if (verbose) {
      std::cout << "    Measuring [" << measurement_id << "]..." << std::flush;
    }

    std::vector<double> valid_times_ms;
    // Pre-allocate space to avoid reallocations during measurement loop.
    valid_times_ms.reserve(samples * iterations_per_sample);
    // Reusable buffer for results to avoid allocation overhead in the loop.
    std::vector<RangeResult> results_buffer;
    results_buffer.reserve(
        config.ranges.size()); // Pre-reserve based on expected output size.

    for (int s = 0; s < samples; ++s) {
      if (verbose)
        std::cout << " S" << (s + 1) << ":";
      int sample_successes = 0;
      for (int iter = 0; iter < iterations_per_sample; ++iter) {
        // Ensure a clean state for each run.
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
          // Log failures during measurement, as they might indicate issues.
          std::cerr
              << "\n      Warning: Execution failed during measurement for ["
              << measurement_id << "], Sample " << s + 1 << ", Iter "
              << iter + 1 << std::endl;
          // Depending on requirements, excessive failures could invalidate the
          // measurement.
        }
        // A short sleep is generally not needed for CPU-bound tasks like
        // Collatz unless thermal throttling on sustained runs is a major
        // concern. It can also slightly skew timing results if used improperly.
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (verbose && iterations_per_sample > 1) {
        // Show success rate per sample if multiple iterations are used.
        std::cout << "(" << sample_successes << "/" << iterations_per_sample
                  << " ok) " << std::flush;
      }
    }

    if (valid_times_ms.empty()) {
      if (verbose)
        std::cout << " FAILED (No successful runs)" << std::endl;
      std::cerr << "\n      Error: Measurement failed for [" << measurement_id
                << "]. No successful iterations completed." << std::endl;
      return -1.0; // Indicate measurement failure.
    }

    // Calculate the median time from the collected valid run times.
    std::sort(valid_times_ms.begin(), valid_times_ms.end());
    size_t n = valid_times_ms.size();
    double median_ms =
        (n % 2 != 0) ? valid_times_ms[n / 2] // Middle element for odd size
                     : (valid_times_ms[n / 2 - 1] + valid_times_ms[n / 2]) /
                           2.0; // Average of two middle elements for even size

    if (verbose) {
      std::cout << " -> Median: " << std::fixed << std::setprecision(4)
                << median_ms << " ms (" << n << " valid runs)" << std::endl;
    }

    return median_ms;
  }
};

/**
 * @brief Manages the execution of benchmark experiments across different
 *        schedulers, workloads, thread counts, and chunk sizes.
 *        Collects results and writes them to a consolidated CSV file.
 */
class ExperimentRunner {
private:
  std::ofstream csv_file;   // Output stream for the results CSV file.
  TimeMeasurer measurer;    // Instance of TimeMeasurer for timing runs.
  std::string csv_filename; // Path to the output CSV file.
  std::vector<std::string>
      workload_descriptions; // Descriptions for each workload.

  // Define the CSV header structure once.
  const std::string CSV_HEADER =
      "WorkloadID,WorkloadDescription,SchedulerName,"
      "SchedulerType,StaticVariant,NumThreads,"
      "ChunkSize,ExecutionTimeMs,BaselineTimeMs,Speedup";

public:
  /**
   * @brief Constructs an ExperimentRunner.
   * @param filename The path for the output CSV file.
   * @param samples The number of samples for the TimeMeasurer.
   * @param iterations The number of iterations per sample for the TimeMeasurer.
   * @param workload_descriptions Descriptions corresponding to each workload.
   */
  ExperimentRunner(std::string filename, int samples, int iterations,
                   const std::vector<std::string> &workload_descriptions)
      : measurer(samples, iterations), csv_filename(std::move(filename)),
        workload_descriptions(workload_descriptions) {
    // Open the CSV file and write the header.
    csv_file = TestUtils::open_csv_file(csv_filename, CSV_HEADER);
  }

  /**
   * @brief Runs the entire benchmark suite.
   *        Iterates through workloads, schedulers, threads, and chunk sizes.
   *
   * @param baseline_schedulable The Schedulable representing the sequential
   * baseline.
   * @param schedulables_to_test Vector of all Schedulables to benchmark
   * (including baseline).
   * @param workloads Vector of workloads, where each workload is a
   * vector<Range>.
   * @param thread_counts Vector of thread counts to test for parallel
   * schedulers.
   * @param chunk_sizes Vector of chunk sizes to test for applicable schedulers.
   * @return true if the suite completed without measurement or file errors,
   *         false otherwise (individual run failures are recorded but don't
   * stop the suite).
   * @note Establishes a separate sequential baseline time for *each* workload
   *       to ensure accurate speedup calculation, as baseline performance can
   *       vary with input data characteristics.
   */
  bool run_suite(const Schedulable &baseline_schedulable,
                 const std::vector<Schedulable> &schedulables_to_test,
                 const std::vector<std::vector<Range>> &workloads,
                 const std::vector<int> &thread_counts,
                 const std::vector<ull> &chunk_sizes) {
    bool overall_success = true; // Tracks if any *measurement* failed.

    if (workloads.size() != workload_descriptions.size()) {
      std::cerr << "Error: Mismatch between number of workloads ("
                << workloads.size() << ") and descriptions ("
                << workload_descriptions.size() << "). Aborting benchmark."
                << std::endl;
      return false;
    }

    // --- Workload Loop ---
    for (size_t workload_idx = 0; workload_idx < workloads.size();
         ++workload_idx) {
      const auto current_workload = workloads[workload_idx];
      const auto current_description = workload_descriptions[workload_idx];
      std::cout << "\n--- Testing Workload " << workload_idx << ": "
                << current_description << " ---" << std::endl;
      // Display ranges for context
      std::cout << "  Ranges: ";
      for (const auto &r : current_workload) {
        std::cout << "[" << r.start << "-" << r.end << "] ";
      }
      std::cout << std::endl;

      // 1. Establish Baseline Time for this specific workload
      std::cout << "  Establishing baseline (Sequential)..." << std::flush;
      Config baseline_config = baseline_schedulable.create_config(
          current_workload, 1, 0); // 1 thread, 0 chunk
      std::string baseline_id =
          "Sequential Baseline W" + std::to_string(workload_idx);
      double baseline_time_ms = measurer.measure_median_time_ms(
          baseline_schedulable.run_func, baseline_config, baseline_id);

      if (baseline_time_ms <= 0) {
        std::cerr << " FAILED. Cannot proceed with benchmarking this workload."
                  << std::endl;
        // Record the baseline failure in the CSV.
        write_result(workload_idx, current_description, baseline_schedulable, 1,
                     0, -1.0, -1.0); // Use -1.0 time to indicate error
        overall_success = false;     // Mark the suite as having issues.
        continue;                    // Skip to the next workload.
      }
      std::cout << " Done. Baseline Time: " << std::fixed
                << std::setprecision(4) << baseline_time_ms << " ms"
                << std::endl;

      // Write the successful baseline result to the CSV (Speedup = 1.0).
      write_result(workload_idx, current_description, baseline_schedulable, 1,
                   0, baseline_time_ms, baseline_time_ms);

      // 2. Test all other Schedulable configurations for this workload
      // --- Scheduler Loop ---
      for (const auto &sched : schedulables_to_test) {
        // Skip the baseline scheduler itself, as it was already measured.
        if (sched.name == baseline_schedulable.name)
          continue;

        // Determine the parameter ranges relevant for this scheduler.
        const std::vector<int> &threads_to_test =
            sched.requires_threads
                ? thread_counts
                : std::vector<int>{1}; // If not parallel, only "run" with T=1
                                       // (conceptually).

        const std::vector<ull> &chunks_to_test =
            sched.requires_chunk_size
                ? chunk_sizes
                : std::vector<ull>{
                      0}; // Use 0 chunk size if not required (represents N/A).

        // --- Thread Loop ---
        for (int t : threads_to_test) {
          // Skip T=1 for explicitly parallel schedulers, as baseline covers it.
          if (sched.requires_threads && t <= 1)
            continue;

          // --- Chunk Size Loop ---
          for (ull c : chunks_to_test) {
            // Skip invalid configurations:
            // - Dynamic scheduler requires chunk size > 0.
            if (sched.type_enum == SchedulingType::DYNAMIC && c == 0)
              continue;
            // - If a scheduler doesn't use chunk size, only run the c=0 (N/A)
            // case once.
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
              // Record successful run result.
              write_result(workload_idx, current_description, sched, t, c,
                           exec_time_ms, baseline_time_ms);
            } else {
              std::cout << " -> FAILED measurement/execution." << std::endl;
              // Record measurement failure in the CSV.
              write_result(workload_idx, current_description, sched, t, c, -1.0,
                           baseline_time_ms); // Time -1.0 indicates failure
              overall_success = false; // Mark the suite as having issues.
            }
          } // end chunk loop
        }   // end thread loop
      }     // end scheduler loop
    }       // end workload loop

    finalize(); // Close the CSV file.
    return overall_success;
  }

  /**
   * @brief Writes a single result row to the CSV file.
   *
   * @param workload_idx Index of the current workload.
   * @param description Description of the workload.
   * @param sched The Schedulable that was run.
   * @param threads Number of threads used.
   * @param chunk_size Chunk size used (0 or >0).
   * @param exec_time_ms Measured median execution time in ms (-1.0 on error).
   * @param baseline_time_ms Baseline sequential time for this workload in ms.
   * @note Calculates speedup based on baseline and execution times.
   *       Handles formatting for CSV, including quoting the description.
   *       Represents chunk size as "N/A" if not applicable or 0.
   */
  void write_result(size_t workload_idx, const std::string &description,
                    const Schedulable &sched, int threads, ull chunk_size,
                    double exec_time_ms, double baseline_time_ms) {
    if (!csv_file.is_open())
      return; // Safety check

    double speedup = 0.0;
    // Calculate speedup only if both times are valid and positive.
    if (exec_time_ms > 0 && baseline_time_ms > 0) {
      speedup = baseline_time_ms / exec_time_ms;
    }

    csv_file << workload_idx << ",";
    // Enclose description in double quotes to handle potential commas within
    // it.
    csv_file << "\"" << description << "\",";
    csv_file << "\"" << sched.name << "\",";
    csv_file << sched.type_str << ",";
    csv_file << sched.variant_str << ",";
    csv_file << threads << ",";
    // Represent chunk size clearly: N/A if not used/required, otherwise the
    // value.
    if (sched.requires_chunk_size && chunk_size > 0) {
      csv_file << chunk_size << ",";
    } else {
      csv_file << "N/A,";
    }

    // Write timing results and speedup, handling potential execution errors.
    if (exec_time_ms > 0) {
      csv_file << std::fixed << std::setprecision(4) << exec_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << baseline_time_ms << ",";
      csv_file << std::fixed << std::setprecision(4) << speedup;
    } else {
      // Indicate error in the execution time field. Baseline is still relevant.
      csv_file << "Error," << std::fixed << std::setprecision(4)
               << baseline_time_ms << ",0.0";
    }
    csv_file << std::endl;
  }

  /**
   * @brief Closes the CSV file stream.
   */
  void finalize() {
    if (csv_file.is_open()) {
      csv_file.close();
      std::cout << "\nBenchmark results saved to " << csv_filename << std::endl;
    }
  }

  /**
   * @brief Destructor ensures the CSV file is closed upon object destruction.
   */
  ~ExperimentRunner() {
    finalize(); // Ensure cleanup happens even if finalize() wasn't explicitly
                // called.
  }
};

// === Schedulable Definitions (Internal Implementation Detail) ===

/**
 * @brief Contains definitions and wrappers for the different schedulers.
 * @note Wrappers adapt specific scheduler functions to the common
 *       `Schedulable::ExecutionFunc` signature required by the framework.
 */
namespace Schedulers {

/**
 * @brief Wrapper for the sequential Collatz calculation.
 *        Adapts `run_sequential` to the `Schedulable::ExecutionFunc` interface.
 * @param cfg Configuration (primarily `cfg.ranges`).
 * @param res Output vector where results will be stored.
 * @return true on success, false if an exception occurs.
 */
bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  try {
    std::vector<ull> seq_results = run_sequential(cfg.ranges);
    res.clear();
    res.reserve(seq_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
      // Create RangeResult with the original range for context.
      res.emplace_back(cfg.ranges[i]);
      // Store the calculated sequential result into the atomic member.
      // Relaxed memory order is fine here as this is the final write after
      // calculation.
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

/**
 * @brief Wrapper for the static scheduling implementations.
 *        Calls `run_static_scheduling`, which internally uses
 * `cfg.static_variant`.
 * @param cfg Configuration containing ranges, thread count, variant, and chunk
 * size.
 * @param res Output vector for results.
 * @return Result of `run_static_scheduling`.
 */
bool run_static_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  // Directly forward the call; run_static_scheduling matches the required
  // signature.
  return run_static_scheduling(cfg, res);
}

/**
 * @brief Wrapper for the dynamic task queue scheduling implementation.
 * @param cfg Configuration containing ranges, thread count, and chunk size.
 * @param res Output vector for results.
 * @return Result of `run_dynamic_work_stealing`.
 */
bool run_dynamic_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  // Directly forward the call; run_dynamic_work_stealing matches the required
  // signature.
  return run_dynamic_work_stealing(cfg, res);
}

// Define the standard set of Schedulable objects to be used in tests.
const Schedulable Sequential("Sequential", "Sequential", "N/A",
                             run_sequential_wrapper, false, false);
const Schedulable StaticBlock(
    "Static Block", "Static", "Block", run_static_wrapper, true,
    false); // Block partitioning doesn't inherently need a 'chunk size' param.
const Schedulable
    StaticCyclic("Static Cyclic", "Static", "Cyclic", run_static_wrapper, true,
                 false); // Pure cyclic distribution doesn't use chunk size.
const Schedulable
    StaticBlockCyclic("Static Block-Cyclic", "Static",
                      "BlockCyclic", // Standardized variant name
                      run_static_wrapper, true,
                      true); // Block-Cyclic requires a chunk size.
const Schedulable Dynamic("Dynamic", "Dynamic", "N/A", run_dynamic_wrapper,
                          true,
                          true); // Dynamic scheduler requires a chunk size > 0.

/** @brief Vector containing all defined schedulers for comprehensive
 * benchmarking. Includes Sequential to have its results alongside others in the
 * CSV. */
const std::vector<Schedulable> AllSchedulers = {
    Sequential, StaticBlock, StaticCyclic, StaticBlockCyclic, Dynamic};

/** @brief Vector containing only the parallel schedulers, useful for
 * correctness tests where comparison against sequential is the goal. */
const std::vector<Schedulable> AllParallelSchedulers = {
    StaticBlock, StaticCyclic, StaticBlockCyclic, Dynamic};

} // namespace Schedulers

// === Correctness Testing Implementation ===

/**
 * @brief Defines a single test case for the correctness suite.
 */
struct CorrectnessTestCase {
  std::string name;               // Name of the test case.
  std::vector<Range> ranges;      // Input ranges for this test.
  std::vector<int> thread_counts; // Set of thread counts to test (>1).
  std::vector<ull> chunk_sizes; // Set of chunk sizes to test (>0 or 0 for N/A).
};

/**
 * @brief Runs the correctness test suite.
 *        Compares the output of parallel schedulers against the sequential
 * version for various workloads, thread counts, and chunk sizes.
 * @return true if all test cases and all configurations within them pass, false
 * otherwise.
 */
bool run_correctness_suite() {
  std::cout << "\n=== Running Correctness Suite ===" << std::endl;
  int total_cases = 0;
  int passed_cases = 0;

  // Define a diverse set of test cases covering edge conditions and typical
  // scenarios.
  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {2, 4}, {1, 8, 32}},
      {"Single Value Range",
       {{27, 27}},
       {2, 4},
       {1}}, // Test single item processing.
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {2, 4, 8},
       {1, 5, 10}},
      {"Medium Range", {{1, 5000}}, {2, 8}, {64, 128}}, // Typical workload
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {2, 4}, {16, 32}},
      {"Empty Range Input",
       {{50, 40}},
       {2, 4},
       {1}}, // start > end, should yield 0 steps.
      {"Minimum Value", {{1, 1}}, {2, 4}, {1}}, // Smallest valid input.
      {"Large Chunk Size",
       {{1, 50}},
       {2, 4},
       {100, 200}}, // Chunk > range size.
      {"More Threads Than Items",
       {{1, 8}},
       {16, 32},
       {1, 2}}, // Test thread oversubscription.
  };

  // --- Test Case Loop ---
  for (const auto &tc : test_cases) {
    total_cases++;
    bool case_passed_overall =
        true; // Tracks if all configurations for this case pass.
    std::cout << "\n[Test Case " << total_cases << ": " << tc.name << "]"
              << std::endl;
    std::cout << "  Ranges: ";
    for (const auto &r : tc.ranges)
      std::cout << "[" << r.start << "-" << r.end << "] ";
    std::cout << std::endl;

    // 1. Generate expected results using the reliable sequential
    // implementation.
    std::cout << "  Generating expected results (Sequential)..." << std::flush;
    std::vector<ull> expected_values;
    bool seq_success = false;
    try {
      // Directly call run_sequential for clarity in correctness testing.
      expected_values = run_sequential(tc.ranges);
      seq_success = true;
      std::cout << " Done (" << expected_values.size() << " results)."
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << " FAILED (Sequential execution error: " << e.what()
                << "). Skipping case." << std::endl;
      case_passed_overall = false; // Mark case as failed if baseline fails.
    } catch (...) {
      std::cerr
          << " FAILED (Unknown sequential execution error). Skipping case."
          << std::endl;
      case_passed_overall = false;
    }

    if (!seq_success) {
      continue; // Move to the next test case if baseline couldn't be generated.
    }

    // 2. Run each parallel scheduler with the specified thread/chunk
    // combinations.
    int sub_test_count = 0;  // Total configurations tested for this case.
    int sub_test_passed = 0; // Configurations passed for this case.

    // --- Parallel Scheduler Loop ---
    for (const auto &sched : Schedulers::AllParallelSchedulers) {

      // Determine parameters for this scheduler based on test case and
      // scheduler requirements.
      const std::vector<int> &threads_to_use =
          tc.thread_counts; // Use threads from test case.

      std::vector<ull> chunks_to_use;
      if (sched.requires_chunk_size) {
        chunks_to_use = tc.chunk_sizes;
        // Dynamic scheduler requires chunk > 0. Remove 0 if present in test
        // case chunks.
        if (sched.type_enum == SchedulingType::DYNAMIC) {
          chunks_to_use.erase(
              std::remove(chunks_to_use.begin(), chunks_to_use.end(), 0),
              chunks_to_use.end());
        }
        // Ensure there's at least one valid chunk size if required.
        if (chunks_to_use.empty()) {
          // Use a default small chunk if the test case accidentally provided
          // none/only 0.
          chunks_to_use.push_back(1);
        }
      } else {
        // If scheduler doesn't use chunk size, test only with the 'N/A' case
        // (represented by 0).
        chunks_to_use = {0};
      }

      // --- Thread Loop ---
      for (int t : threads_to_use) {
        // Correctness tests focus on parallel execution (T > 1).
        if (t <= 1)
          continue;

        // --- Chunk Size Loop ---
        for (ull c : chunks_to_use) {
          // Double-check: Skip chunk 0 for Dynamic (should have been handled
          // above, but safety).
          if (sched.type_enum == SchedulingType::DYNAMIC && c == 0)
            continue;
          // Skip non-zero chunks if the scheduler doesn't require them.
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

          // Execute the scheduler function within a try-catch block.
          try {
            run_success = sched.run_func(run_config, actual_results);
          } catch (const std::exception &e) {
            run_success = false;
            error_msg = " Exception: " + std::string(e.what());
          } catch (...) {
            run_success = false;
            error_msg = " Unknown exception";
          }

          // Compare results if execution was successful.
          bool result_match = false;
          if (run_success) {
            result_match =
                TestUtils::compare_results(expected_values, actual_results,
                                           test_id, true /* verbose error */);
          }

          // Report PASS/FAIL for this specific configuration.
          if (run_success && result_match) {
            std::cout << " PASS" << std::endl;
            sub_test_passed++;
          } else {
            std::cout << " FAIL";
            if (!run_success) {
              std::cout << " (Execution Error)." << error_msg;
            } else { // implies !result_match
              std::cout << " (Result Mismatch).";
              // compare_results already printed the details if verbose was
              // true.
            }
            std::cout << std::endl;
            case_passed_overall = false; // Mark the entire test case as failed
                                         // if any config fails.
          }
        } // end chunk loop
      }   // end thread loop
    }     // end scheduler loop

    std::cout << "  Case Summary: " << sub_test_passed << "/" << sub_test_count
              << " parallel configurations passed." << std::endl;

    if (case_passed_overall) {
      passed_cases++;
    }
  } // end test case loop

  // Print the final summary of the correctness suite.
  std::cout << "\n=== Correctness Suite Summary ===" << std::endl;
  TestUtils::print_summary_line("Correctness Cases", total_cases, passed_cases);
  std::cout << "===================================" << std::endl;
  // Return true only if all defined test cases passed completely.
  return (total_cases > 0 && total_cases == passed_cases);
}

// === Performance Benchmark Suite Implementation ===

/**
 * @brief Runs the main performance benchmark suite using the ExperimentRunner.
 *
 * @param thread_counts Vector of thread counts to benchmark.
 * @param chunk_sizes Vector of chunk sizes to benchmark.
 * @param workloads Vector of different workloads (sets of ranges) to test.
 * @param workload_descriptions Corresponding descriptions for each workload.
 * @param samples Number of samples for time measurement.
 * @param iterations_per_sample Iterations per sample for time measurement.
 * @return true if the benchmark suite ran to completion (even with individual
 *         run failures recorded), false if a fatal error occurred (e.g., file
 * I/O).
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
    // Initialize the ExperimentRunner.
    ExperimentRunner runner(BenchmarkConfig::BENCHMARK_CSV_FILE, samples,
                            iterations_per_sample, workload_descriptions);

    // Execute the full suite.
    // Pass the designated baseline scheduler and the list of all schedulers to
    // test.
    bool success =
        runner.run_suite(Schedulers::Sequential,    // Define the baseline
                         Schedulers::AllSchedulers, // Schedulers to benchmark
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
    // Return the overall status (true if runner completed, false on fatal
    // error).
    return success;

  } catch (const std::exception &e) {
    // Catch potential fatal errors during setup or file operations.
    std::cerr << "\nFATAL ERROR during benchmark execution: " << e.what()
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "\nFATAL UNKNOWN ERROR during benchmark execution."
              << std::endl;
    return false;
  }
}
