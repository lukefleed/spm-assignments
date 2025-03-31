#include "common_types.h"      // Core data types (Range, Config, etc.)
#include "dynamic_scheduler.h" // Declaration for run_dynamic_task_queue
#include "sequential.h"        // Declaration for run_sequential
#include "static_scheduler.h"  // Declaration for run_static_scheduling
#include "testing.h"           // Benchmark and correctness test suites
#include "utils.h"             // Utilities like Timer and argument parsing
#include <chrono>              // For potential timing
#include <cstdlib>             // For EXIT_SUCCESS, EXIT_FAILURE
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
    return "Dynamic Task Queue";
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
      // Delegate to the dynamic scheduling function.
      success = run_dynamic_task_queue(config, results);
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
 * @brief Checks if the program was invoked with a flag indicating a special
 * mode (correctness testing or benchmarking).
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return true if the first argument matches a known test/benchmark flag, false
 * otherwise.
 * @note Using std::string_view avoids unnecessary string allocation for
 * comparison.
 */
[[nodiscard]] bool is_test_or_benchmark_mode(int argc, char *argv[]) {
  if (argc < 2) {
    return false; // Not enough arguments for a flag.
  }
  // Efficiently compare the first argument without creating a std::string.
  const std::string_view first_arg(argv[1]);
  return first_arg == AppConstants::TEST_CORRECTNESS_FLAG ||
         first_arg == AppConstants::BENCHMARK_FLAG;
}

/**
 * @brief Handles the execution flow when a test or benchmark flag is detected.
 *        Parses the specific flag and runs the corresponding suite.
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @return true if the specified suite ran successfully, false on failure or
 * unknown flag.
 * @note Benchmark parameters (threads, chunks, workloads) are currently
 * hardcoded within this function but could be made configurable via additional
 * command-line args.
 */
[[nodiscard]] bool handle_test_or_benchmark_mode([[maybe_unused]] int argc,
                                                 char *argv[]) {
  // This function assumes argc >= 2 based on the caller
  // (is_test_or_benchmark_mode).
  const std::string_view first_arg(argv[1]);

  if (first_arg == AppConstants::TEST_CORRECTNESS_FLAG) {
    std::cout << "Running Correctness Test Suite..." << std::endl;
    // Delegate execution to the correctness suite function from testing.h.
    return run_correctness_suite();
  }

  if (first_arg == AppConstants::BENCHMARK_FLAG) {
    std::cout << "Running Performance Benchmark Suite..." << std::endl;

    // --- Benchmark Configuration ---
    // These parameters define the scope of the performance benchmark.

    // Determine thread counts to test, scaling up to available hardware cores.
    const int max_threads = std::thread::hardware_concurrency();
    std::vector<int> threads_to_test;
    // Start from 2 threads for parallel benchmarks; 1 thread is the sequential
    // baseline handled automatically by the ExperimentRunner. A linear scaling
    // strategy is used here for simplicity. Other strategies (e.g., powers of
    // 2, specific core counts) could be employed depending on the machine
    // architecture and testing goals.
    for (int i = 2; i <= max_threads; ++i) {
      threads_to_test.push_back(i);
    }
    // Ensure the maximum hardware concurrency is always included if > 1.
    if (max_threads > 1 &&
        (threads_to_test.empty() || threads_to_test.back() != max_threads)) {
      threads_to_test.push_back(max_threads);
    }
    if (max_threads <= 1) {
      // Inform the user if parallelism is limited by hardware.
      std::cout << "Warning: Only " << max_threads
                << " hardware thread(s) detected. "
                << "Parallel benchmarks might not show significant speedup."
                << std::endl;
      if (max_threads == 1 && threads_to_test.empty()) {
        // Add 2 threads anyway to test overhead if user insists on parallel run
        // threads_to_test.push_back(2); // Or just rely on sequential baseline.
        // Let's stick to baseline.
      }
    }

    // Define chunk sizes to test for relevant schedulers (Static Block-Cyclic,
    // Dynamic). This selection covers a range from small (potentially
    // cache-friendly) to large. The optimal chunk size often depends on
    // workload characteristics, cache sizes, and scheduling overhead. Testing a
    // range helps identify this sensitivity.
    const std::vector<ull> chunks_to_test = {16, 32, 64, 128, 256, 512, 1024};

    // --- Benchmark Workload Configuration ---

    // Helper functions to create more complex workloads
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

    // Helper function to create ranges around powers of 2
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

    // Helper function to create extreme imbalance with isolated expensive
    // This should be where the dynamic scheduler shines
    auto create_extreme_imbalance = []() {
      std::vector<Range> ranges;

      // Basic range mix
      ranges.push_back({1, 10000});         // Mostly cheap calculations
      ranges.push_back({2000000, 2001000}); // Medium cost

      // Very expensive isolated calculations
      for (ull i = 0; i < 100; i++) {
        ull expensive_start = 100000000 + (i * 5000000);
        ranges.push_back({expensive_start, expensive_start + 5});
      }

      // Known difficult numbers in isolated ranges
      const std::vector<ull> difficult_numbers = {27,    73,     9663,
                                                  77671, 837799, 8400511};

      for (ull num : difficult_numbers) {
        ranges.push_back({num, num});
      }

      return ranges;
    };

    // Define workload pairs (workload and its description)
    struct WorkloadPair {
      std::vector<Range> ranges;
      std::string description;
    };

    const std::vector<WorkloadPair> workload_pairs = {
        {{{1, 100000}}, "Medium Balanced (1-100k)"},

        {{{1, 1000000}}, "Large Balanced (1-1M)"},

        {{{1, 100}, {10000, 501000}, {5000, 50000}},
         "Unbalanced Mix (Small, Large, Medium)"},

        {create_small_uniform_ranges(), "Many Small Uniform Ranges (500x1k)"},

        // {{{9663, 9663}, {77671, 77671}, {626331, 626331}, {837799, 837799}},
        //  "Known High-Step Points"},

        {create_power_of_two_ranges(),
         "Ranges Around Powers of 2 (2^8 to 2^20)"},

        {create_extreme_imbalance(),
         "Extreme Imbalance with Isolated Expensive Calculations"}};

    // Extract workloads and descriptions for the benchmark suite
    std::vector<std::vector<Range>> workloads;
    std::vector<std::string> workload_descriptions;

    for (const auto &pair : workload_pairs) {
      workloads.push_back(pair.ranges);
      workload_descriptions.push_back(pair.description);
    }

    // Parameters for the TimeMeasurer (used by ExperimentRunner).
    // Higher values yield more statistically robust results but increase
    // benchmark duration.
    // - Samples: Independent repetitions of the measurement process.
    // - Iterations per sample: Runs within a sample to mitigate cold start
    // effects and variability.
    const int samples = 10;               // Number of measurement samples.
    const int iterations_per_sample = 20; // Runs per sample.

    // Delegate execution to the benchmark suite function from testing.h.
    return run_benchmark_suite(threads_to_test, chunks_to_test, workloads,
                               workload_descriptions, samples,
                               iterations_per_sample);
  }

  // If the flag is not recognized (should not happen if called after
  // is_test_or_benchmark_mode).
  std::cerr << "Error: Unrecognized flag '" << first_arg << "'." << std::endl;
  return false;
}

/**
 * @brief Main application entry point.
 *        Determines the execution mode (normal, test, benchmark) based on
 * arguments, parses configuration, executes the appropriate logic, and reports
 * results.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE otherwise.
 */
int main(int argc, char *argv[]) {
  // First, check if a special mode (test or benchmark) is requested.
  if (is_test_or_benchmark_mode(argc, argv)) {
    bool success = handle_test_or_benchmark_mode(argc, argv);
    // Exit based on the success status of the test/benchmark suite.
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // --- Normal Execution Mode ---
  // If no special flag is given, proceed with standard calculation.

  // Parse command-line arguments into a Config struct.
  // parse_arguments handles usage instructions and error reporting.
  auto config_opt = parse_arguments(argc, argv);
  if (!config_opt) {
    // Argument parsing failed, error message already printed by
    // parse_arguments.
    return EXIT_FAILURE;
  }
  Config config =
      *config_opt; // Dereference the optional to get the Config object.

  std::vector<RangeResult> results; // Vector to store calculation results.
  Timer timer;                      // Timer to measure execution duration.

  // Execute the Collatz calculation based on the parsed configuration.
  bool success = execute_collatz_calculation(config, results);
  double elapsed_time_s =
      timer.elapsed_s(); // Get elapsed time after execution.

  if (success) {
    // Print results and statistics if calculation was successful.
    print_results(results, config, elapsed_time_s);
    return EXIT_SUCCESS;
  } else {
    // Error message should have been printed by execute_collatz_calculation.
    std::cerr << "Computation failed. See previous errors for details."
              << std::endl;
    return EXIT_FAILURE;
  }
}
