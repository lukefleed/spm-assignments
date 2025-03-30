#include "common_types.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "testing.h"
#include "utils.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Wrapper function for sequential execution mode
 *
 * This function encapsulates the sequential algorithm execution to provide
 * a consistent interface with the parallel implementations.
 *
 * @param config Configuration parameters
 * @param results_out Vector to store computation results
 * @return true if execution completed successfully, false otherwise
 */
bool run_sequential_wrapper(const Config &config,
                            std::vector<RangeResult> &results_out) {
  try {
    // Run the sequential implementation
    std::vector<ull> seq_results = run_sequential(config.ranges);

    // Convert results format
    results_out.clear();
    results_out.reserve(seq_results.size());

    for (size_t i = 0; i < seq_results.size(); ++i) {
      RangeResult rr(config.ranges[i]);
      rr.max_steps.store(seq_results[i]);
      results_out.push_back(std::move(rr));
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Sequential execution failed: " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Execute test suites based on command line arguments
 *
 * @param argc Command line argument count
 * @param argv Command line argument values
 * @return int Exit code (0 for success, 1 for failure)
 */
bool handle_test_mode(int argc, char *argv[]) {
  if (argc < 2)
    return false;

  std::string first_arg = argv[1];

  if (first_arg == "--test-correctness") {
    // Run comprehensive correctness tests
    return run_correctness_suite();
  } else if (first_arg == "--test-performance") {
    // Performance test configuration
    std::vector<Range> perf_workload = {{1, 1000}, {1000000, 2000000}};

    // Test all available threads to evaluate scalability
    int max_threads = std::thread::hardware_concurrency();
    std::vector<int> threads_to_test;
    for (int i = 1; i <= max_threads; ++i) {
      threads_to_test.push_back(i);
    }

    // Test different chunk sizes to evaluate performance impact
    std::vector<ull> chunks_to_test = {64, 128, 256};

    // Statistical parameters for reliable measurements
    int samples = 2;               // Number of median measurements
    int iterations_per_sample = 2; // Executions per measurement

    return run_performance_suite(threads_to_test, chunks_to_test, samples,
                                 iterations_per_sample, perf_workload);
  } else if (first_arg == "--test-static-performance") {
    // Static scheduler variants performance test configuration
    std::vector<Range> static_workload = {{1, 1000}, {1000000, 2000000}};

    int max_threads = std::thread::hardware_concurrency();
    std::vector<int> threads_to_test;
    for (int i = 1; i <= max_threads; ++i) {
      threads_to_test.push_back(i);
    }

    // Test more granular chunk sizes for static scheduling analysis
    std::vector<ull> chunks_to_test = {16, 32, 64, 128, 256};

    int samples = 2;
    int iterations_per_sample = 2;

    return run_static_performance_comparison(threads_to_test, chunks_to_test,
                                             samples, iterations_per_sample,
                                             static_workload);
  } else if (first_arg == "--test-workload-scaling") {
    // Define various workloads with different characteristics
    std::vector<std::vector<Range>> workloads = {
        // Workload 1: Small range with even distribution
        {{1, 10000}},

        // Workload 2: Large range
        {{1, 1000000}},

        // Workload 3: Multiple small ranges
        {{1, 1000}, {2000, 3000}, {4000, 5000}, {6000, 7000}},

        // Workload 4: Uneven distribution (some ranges much larger)
        {{1, 100}, {1000, 100000}},

        // Workload 5: Very small range (minimal work)
        {{1, 100}}};

    // Select thread counts for scaling test
    std::vector<int> thread_counts;
    int max_threads = std::thread::hardware_concurrency();

    // Use 1, 2, 3 ... up to max_threads for scaling test
    for (int t = 1; t <= max_threads; t += 1) {
      thread_counts.push_back(t);
    }

    int samples = 2;
    int iterations_per_sample = 2;

    return run_workload_scaling_tests(thread_counts, workloads, samples,
                                      iterations_per_sample);
  }

  return false;
}

/**
 * @brief Get the scheduler type name as string
 *
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
    }
    return "Static " + variant_name;
  } else {
    return "Dynamic Task Queue";
  }
}

/**
 * @brief Execute the appropriate Collatz calculation based on configuration
 *
 * @param config Application configuration
 * @param results Output vector for calculation results
 * @return true on successful execution, false otherwise
 */
bool execute_collatz_calculation(const Config &config,
                                 std::vector<RangeResult> &results) {
  std::string scheduler_name = get_scheduler_name(config);

  if (config.verbose) {
    std::cout << "Running " << scheduler_name << " scheduler..." << std::endl;
  }

  if (config.num_threads == 1) {
    // Use sequential implementation for single-threaded execution
    // This optimizes the single-thread case by avoiding thread creation
    // overhead
    return run_sequential_wrapper(config, results);
  } else if (config.scheduling == SchedulingType::STATIC) {
    // Use static scheduling for predictable workloads with known distribution
    return run_static_scheduling(config, results);
  } else {
    // Use dynamic scheduling for better load balancing with irregular workloads
    return run_dynamic_task_queue(config, results);
  }
}

/**
 * @brief Print calculation results and execution statistics
 *
 * @param results Calculation results to display
 * @param config Application configuration
 * @param elapsed_time_s Execution time in seconds
 */
void print_results(const std::vector<RangeResult> &results,
                   const Config &config, double elapsed_time_s) {
  // Print calculation results in the required format
  for (const auto &res : results) {
    std::cout << res.original_range.start << "-" << res.original_range.end
              << ": " << res.max_steps.load() << std::endl;
  }

  // Print performance statistics when verbose mode is enabled
  if (config.verbose) {
    std::string scheduler_name = get_scheduler_name(config);

    std::cout << "\nTotal execution time: " << std::fixed
              << std::setprecision(4) << elapsed_time_s << " seconds"
              << std::endl;
    std::cout << "Using " << config.num_threads << " threads." << std::endl;
    std::cout << "Scheduling: " << scheduler_name;

    // Print chunk size for applicable schedulers
    if ((config.scheduling == SchedulingType::STATIC &&
         config.num_threads > 1) ||
        config.scheduling == SchedulingType::DYNAMIC) {
      std::cout << ", Chunk Size: " << config.chunk_size;
    }
    std::cout << std::endl;
  }
}

/**
 * @brief Main application entry point
 *
 * Parses command line arguments, executes the appropriate scheduler,
 * and displays results.
 *
 * @param argc Command line argument count
 * @param argv Command line argument values
 * @return int Exit code (0 for success, 1 for failure)
 */
int main(int argc, char *argv[]) {
  // Handle test mode if test flags are present
  if (argc >= 2) {
    bool is_test = argv[1][0] == '-' && argv[1][1] == '-';
    if (is_test) {
      bool success = handle_test_mode(argc, argv);
      return success ? 0 : 1;
    }
  }

  // Normal execution mode - parse arguments
  auto config_opt = parse_arguments(argc, argv);
  if (!config_opt) {
    return 1;
  }
  Config config = *config_opt;

  // Execute the calculation
  std::vector<RangeResult> results;
  Timer timer;
  bool success = execute_collatz_calculation(config, results);
  double elapsed_time_s = timer.elapsed_s();

  // Print results and exit
  if (success) {
    print_results(results, config, elapsed_time_s);
    return 0;
  } else {
    std::cerr << "Error during computation." << std::endl;
    return 1;
  }
}
