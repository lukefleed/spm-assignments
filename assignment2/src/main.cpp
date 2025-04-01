#include "common_types.h"         // Core data types (Range, Config, etc.)
#include "dynamic_scheduler.h"    // Declaration for run_dynamic_work_stealing
#include "sequential.h"           // Declaration for run_sequential
#include "static_scheduler.h"     // Declaration for run_static_scheduling
#include "testing.h"              // Benchmark and correctness test suites
#include "theoretical_analysis.h" // Theoretical analysis functions
#include "utils.h"                // Utilities like Timer and argument parsing
#include <algorithm>              // For std::find
#include <chrono>                 // For potential timing
#include <cstdlib>                // For EXIT_SUCCESS, EXIT_FAILURE
#include <iomanip>     // For formatted output (std::setprecision, std::fixed)
#include <iostream>    // For console input/output (std::cout, std::cerr)
#include <string>      // For std::string
#include <string_view> // For efficient string comparisons without copying
#include <thread>      // For std::thread::hardware_concurrency
#include <vector>      // For std::vector

// --- Constants for Command Line Arguments ---
namespace AppConstants {
/** @brief Command-line flag to trigger the correctness test suite. */
constexpr const char *TEST_CORRECTNESS_FLAG = "--test-correctness";
/** @brief Command-line flag to trigger the performance benchmark suite. */
constexpr const char *BENCHMARK_FLAG = "--benchmark";
/** @brief Command-line flag to trigger the theoretical analysis. */
constexpr const char *THEORY_FLAG = "--theory";
/** @brief Short command-line flag for theoretical analysis. */
constexpr const char *THEORY_FLAG_SHORT = "-t";
} // namespace AppConstants

/**
 * @brief Get a descriptive string name for the scheduler specified in the
 * Config. Used primarily for non-benchmark console output.
 * @param config The application configuration containing scheduler settings.
 * @return std::string A human-readable name of the configured scheduler.
 */
std::string get_scheduler_name(const Config &config) {
  // Handle the sequential case based on thread count first.
  if (config.num_threads == 1) {
    return "Sequential";
  }
  // Handle parallel schedulers.
  else if (config.scheduling == SchedulingType::STATIC) {
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
      break; // Defensive default
    }
    return "Static " + variant_name;
  } else if (config.scheduling == SchedulingType::DYNAMIC) {
    return "Dynamic Work Stealing";
  } else {
    // Should not happen with proper validation, but handle defensively.
    return "Unknown Scheduler Type";
  }
}

/**
 * @brief Executes the Collatz calculation using the scheduler specified in the
 * configuration. This function centralizes the dispatch logic for normal
 * execution mode.
 * @param config The application configuration specifying ranges, threads,
 * scheduling, etc.
 * @param results Output vector where RangeResult objects will be stored.
 * @return true if the calculation completes successfully, false otherwise.
 * @note This function is intended for the standard execution path, *not* for
 * benchmarking or correctness testing, which use dedicated functions in
 * `testing.h`.
 */
bool execute_collatz_calculation(const Config &config,
                                 std::vector<RangeResult> &results) {
  std::string scheduler_name = get_scheduler_name(config);
  bool success = false;

  if (config.verbose) {
    std::cout << "Running " << scheduler_name << " scheduler..." << std::endl;
  }

  try {
    // Dispatch to the appropriate implementation based on configuration.
    if (config.num_threads == 1) {
      // Execute sequentially directly.
      std::vector<ull> seq_results = run_sequential(config.ranges);
      // Convert sequential results (vector<ull>) to the common RangeResult
      // format.
      results.clear();
      results.reserve(seq_results.size());
      for (size_t i = 0; i < seq_results.size(); ++i) {
        results.emplace_back(config.ranges[i]); // Keep original range info
        // Store result using relaxed memory order, sufficient for
        // single-threaded case.
        results.back().max_steps.store(seq_results[i],
                                       std::memory_order_relaxed);
      }
      success = true;
    } else if (config.scheduling == SchedulingType::STATIC) {
      // Delegate to the static scheduling function.
      success = run_static_scheduling(config, results);
    } else if (config.scheduling == SchedulingType::DYNAMIC) {
      success = run_dynamic_work_stealing(config, results);
    } else {
      // This case should ideally be prevented by argument parsing validation.
      std::cerr << "Error: Unknown scheduling type configured." << std::endl;
      success = false;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during " << scheduler_name << " execution: " << e.what()
              << std::endl;
    success = false;
  } catch (...) {
    // Catch any other unexpected exceptions.
    std::cerr << "Unknown error during " << scheduler_name << " execution."
              << std::endl;
    success = false;
  }
  return success;
}

/**
 * @brief Prints the calculated Collatz results and execution statistics (if
 * verbose). Used only for the normal execution mode.
 * @param results The vector of RangeResult containing the calculation outputs.
 * @param config The application configuration used for the run.
 * @param elapsed_time_s The total execution time in seconds.
 */
void print_results(const std::vector<RangeResult> &results,
                   const Config &config, double elapsed_time_s) {
  // Output the max steps found for each original range.
  std::cout << "\n--- Calculation Results ---" << std::endl;
  for (const auto &res : results) {
    std::cout << "Range [" << res.original_range.start << " - "
              << res.original_range.end << "]: Max steps = "
              << res.max_steps.load(std::memory_order_relaxed) << std::endl;
    // Relaxed memory order is sufficient here as we are just reading final
    // results after all computations and thread synchronization have completed.
  }

  // Print detailed performance statistics if verbose mode is enabled.
  if (config.verbose) {
    std::string scheduler_name = get_scheduler_name(config);
    std::cout << "\n--- Execution Summary ---" << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(4)
              << elapsed_time_s << " seconds" << std::endl;
    std::cout << "Threads used: " << config.num_threads << std::endl;
    std::cout << "Scheduling: " << scheduler_name;
    // Only show chunk size if relevant (parallel execution and scheduler uses
    // it).
    if (config.num_threads > 1 &&
        (config.scheduling == SchedulingType::STATIC ||
         config.scheduling == SchedulingType::DYNAMIC) &&
        config.chunk_size > 0) // Only display if explicitly set and used
    {
      std::cout << ", Chunk Size: " << config.chunk_size;
    }
    std::cout << std::endl;
    std::cout << "------------------------" << std::endl;
  }
}

/**
 * @brief Generates theoretical speedup data and writes it to a CSV file.
 *        (This just calls the function from theoretical_analysis.h)
 * @param workloads The workloads to analyze.
 * @param workload_descriptions Descriptions of the workloads.
 * @param output_filename The name of the output CSV file.
 * @return true if the analysis completes successfully, false otherwise.
 */
bool run_theoretical_analysis(
    const std::vector<std::vector<Range>> &workloads,
    const std::vector<std::string> &workload_descriptions,
    const std::string &output_filename) {
  std::cout << "\n=== Running Theoretical Analysis ===" << std::endl;
  // Assumes generate_theoretical_speedup_csv is declared in
  // theoretical_analysis.h
  return generate_theoretical_speedup_csv(workloads, workload_descriptions,
                                          output_filename);
}

/**
 * @brief Checks if the program was invoked with a flag indicating a special
 * mode (correctness testing, benchmarking, or theory).
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return true if the first argument matches a known special mode flag, false
 * otherwise.
 * @note Using std::string_view avoids unnecessary string allocation for
 * comparison.
 */
[[nodiscard]] bool is_special_mode(int argc, char *argv[]) {
  if (argc < 2) {
    return false; // Not enough arguments for a flag.
  }
  // Efficiently compare the first argument without creating a std::string.
  const std::string_view first_arg(argv[1]);
  return first_arg == AppConstants::TEST_CORRECTNESS_FLAG ||
         first_arg == AppConstants::BENCHMARK_FLAG ||
         first_arg == AppConstants::THEORY_FLAG ||
         first_arg == AppConstants::THEORY_FLAG_SHORT;
}

/**
 * @brief Handles the execution flow when a special mode flag is detected.
 * Parses the specific flag and runs the corresponding suite or analysis.
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return true if the specified operation ran successfully, false on failure or
 * unknown flag.
 * @note Benchmark parameters (threads, chunks, workloads) are defined here.
 */
[[nodiscard]] bool handle_special_mode([[maybe_unused]] int argc,
                                       char *argv[]) {
  // This function assumes argc >= 2 based on the caller (is_special_mode).
  const std::string_view first_arg(argv[1]);

  // --- Define Workloads (Keep these definitions accessible) ---
  struct WorkloadPair {
    std::vector<Range> ranges;
    std::string description;
  };

  auto create_small_uniform_ranges = []() {
    std::vector<Range> ranges;
    const ull num_ranges = 500;
    const ull range_size = 1000;
    for (ull i = 0; i < num_ranges; ++i) {
      ull start = 1 + (i * range_size);
      ranges.push_back({start, start + range_size - 1});
    }
    return ranges;
  };

  auto create_power_of_two_ranges = []() {
    std::vector<Range> ranges;
    const ull range_width = 1000;
    for (int power = 8; power <= 20; ++power) {
      ull center = 1ULL << power;
      ull start = (center > range_width / 2) ? (center - range_width / 2) : 1;
      ranges.push_back({start, center + range_width / 2});
    }
    return ranges;
  };

  auto create_extreme_imbalance = []() {
    std::vector<Range> ranges;
    ranges.push_back({1, 10000});         // Mostly cheap
    ranges.push_back({2000000, 2001000}); // Medium
    for (ull i = 0; i < 100; i++) {       // Many tiny expensive tasks
      ull expensive_start = 100000000 + (i * 5000000);
      ranges.push_back({expensive_start, expensive_start + 5});
    }
    const std::vector<ull> difficult_numbers = {27,    73,     9663,
                                                77671, 837799, 8400511};
    for (ull num : difficult_numbers) {
      ranges.push_back({num, num});
    }
    return ranges;
  };

  const std::vector<WorkloadPair> workload_pairs = {
      {{{1, 100000}}, "Medium Balanced (1-100k)"},
      {{{1, 1000000}}, "Large Balanced (1-1M)"},
      {{{1, 100}, {10000, 501000}, {5000, 50000}}, "Unbalanced Mix"},
      {create_small_uniform_ranges(), "Many Small Uniform (500x1k)"},
      {create_power_of_two_ranges(), "Ranges Around Powers of 2"},
      {create_extreme_imbalance(), "Extreme Imbalance"}};
  // --- End Workload Definitions ---

  if (first_arg == AppConstants::TEST_CORRECTNESS_FLAG) {
    std::cout << "Running Correctness Test Suite..." << std::endl;
    return run_correctness_suite();
  }

  if (first_arg == AppConstants::BENCHMARK_FLAG) {
    std::cout << "Running Performance Benchmark Suite..." << std::endl;

    const int max_threads_detected = std::thread::hardware_concurrency();
    int max_threads_to_run =
        std::max(2, max_threads_detected); // Ensure at least 2 threads are
                                           // tested if possible

    std::vector<int> threads_to_test;
    // Start from 2 threads for parallel benchmarks
    for (int i = 2; i <= max_threads_to_run; i += 1) {
      threads_to_test.push_back(i);
    }
    // Ensure the maximum detected hardware concurrency is included
    if (threads_to_test.empty() ||
        threads_to_test.back() < max_threads_to_run) {
      threads_to_test.push_back(max_threads_to_run);
    }
    // Remove duplicates just in case
    std::sort(threads_to_test.begin(), threads_to_test.end());
    threads_to_test.erase(
        std::unique(threads_to_test.begin(), threads_to_test.end()),
        threads_to_test.end());

    if (max_threads_detected <= 1) {
      std::cout << "Warning: Only " << max_threads_detected
                << " hardware thread(s) detected. "
                << "Parallel benchmarks might not show significant speedup."
                << std::endl;
      // If only 1 core, maybe just test with 1 or 2 threads?
      // Let's stick to testing with 2 for consistency of parallel logic,
      // but the warning is important.
      if (threads_to_test.empty())
        threads_to_test.push_back(2);
    }

    // Refined chunk sizes, focusing on powers of 2 which are common.
    const std::vector<ull> chunks_to_test = {32, 64, 128, 256, 512, 1024, 2048};

    std::vector<std::vector<Range>> workloads;
    std::vector<std::string> workload_descriptions;
    workloads.reserve(workload_pairs.size());
    workload_descriptions.reserve(workload_pairs.size());
    for (const auto &pair : workload_pairs) {
      workloads.push_back(pair.ranges);
      workload_descriptions.push_back(pair.description);
    }

    const int samples = 10;
    const int iterations_per_sample = 50;

    return run_benchmark_suite(threads_to_test, chunks_to_test, workloads,
                               workload_descriptions, samples,
                               iterations_per_sample);
  }

  if (first_arg == AppConstants::THEORY_FLAG ||
      first_arg == AppConstants::THEORY_FLAG_SHORT) {
    std::vector<std::vector<Range>> workloads;
    std::vector<std::string> workload_descriptions;
    workloads.reserve(workload_pairs.size());
    workload_descriptions.reserve(workload_pairs.size());
    for (const auto &pair : workload_pairs) {
      workloads.push_back(pair.ranges);
      workload_descriptions.push_back(pair.description);
    }

    std::string output_file = "results/theoretical_speedup.csv";
    if (run_theoretical_analysis(workloads, workload_descriptions,
                                 output_file)) {
      std::cout << "Theoretical analysis complete. Results saved to: "
                << output_file << std::endl;
      return true;
    } else {
      std::cerr << "Error running theoretical analysis." << std::endl;
      return false;
    }
  }

  std::cerr << "Error: Unrecognized flag '" << first_arg << "'." << std::endl;
  return false;
}

/**
 * @brief Main application entry point.
 *        Determines the execution mode (normal, test, benchmark, theory) based
 * on arguments, parses configuration, executes the appropriate logic, and
 * reports results.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE otherwise.
 */
int main(int argc, char *argv[]) {
  // Check for special execution modes first.
  if (is_special_mode(argc, argv)) {
    bool success = handle_special_mode(argc, argv);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // --- Normal Execution Mode ---
  auto config_opt = parse_arguments(argc, argv);
  if (!config_opt) {
    // Argument parsing failed or help requested.
    // parse_arguments handles printing usage/errors.
    return EXIT_FAILURE; // Use failure code if config parsing fails.
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
    std::cerr << "Computation failed. See previous errors for details."
              << std::endl;
    return EXIT_FAILURE;
  }
}
