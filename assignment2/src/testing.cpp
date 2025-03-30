#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// File paths for CSV output
const std::string RESULTS_DIR = "results/";
const std::string STATIC_COMPARISON_CSV =
    RESULTS_DIR + "static_performance_data.csv";
const std::string ALL_SCHEDULERS_CSV = RESULTS_DIR + "performance_data.csv";
const std::string WORKLOAD_SCALING_CSV =
    RESULTS_DIR + "workload_scaling_data.csv";

/**
 * @brief Compares expected sequential results with results from a scheduler.
 */
bool compare_results(const std::vector<ull> &expected,
                     const std::vector<RangeResult> &results,
                     const std::string &schedulerType, int threadCount,
                     ull chunkSize) {
  if (results.size() != expected.size()) {
    std::cerr << "  [" << schedulerType << " T=" << threadCount
              << ", C=" << chunkSize << "] Error: expected " << expected.size()
              << " results, got " << results.size() << std::endl;
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (results[i].max_steps.load() != expected[i]) {
      std::cerr << "  [" << schedulerType << " T=" << threadCount
                << ", C=" << chunkSize << "] Mismatch on range " << i
                << ": expected " << expected[i] << ", got "
                << results[i].max_steps.load() << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief Prints summary of test results.
 */
void print_summary_line(const std::string &testName, int total, int passed) {
  std::cout << std::setw(25) << std::left << testName
            << " Total: " << std::setw(4) << total
            << " Passed: " << std::setw(4) << passed
            << " Failed: " << std::setw(4) << (total - passed) << std::endl;
}

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
 * @brief Tests a specific static scheduling variant.
 */
bool test_static_variant(StaticVariant variant, int n_threads, ull chunk_size,
                         const std::vector<Range> &ranges,
                         const std::vector<ull> &expected_results) {
  Config config;
  config.scheduling = SchedulingType::STATIC;
  config.static_variant = variant;
  config.num_threads = n_threads;
  config.chunk_size = chunk_size;
  config.ranges = ranges;
  config.verbose = false;

  std::string variant_name;
  switch (variant) {
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

  std::cout << "  [Static " << variant_name << " T=" << n_threads
            << ", C=" << chunk_size << "] Running..." << std::flush;
  std::vector<RangeResult> static_results;
  bool static_success = run_static_scheduling(config, static_results);
  if (static_success &&
      compare_results(expected_results, static_results,
                      "Static " + variant_name, n_threads, chunk_size)) {
    std::cout << " PASS" << std::endl;
    return true;
  } else {
    std::cout << " FAIL" << std::endl;
    return false;
  }
}

/**
 * @brief Wrapper for sequential implementation.
 */
bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  std::vector<ull> seq_results = run_sequential(cfg.ranges);
  res.clear();
  res.reserve(seq_results.size());
  for (size_t i = 0; i < seq_results.size(); ++i) {
    RangeResult rr(cfg.ranges[i]);
    rr.max_steps.store(seq_results[i]);
    res.push_back(rr);
  }
  return true;
}

/**
 * @brief Measures median execution time of a function.
 */
double measure_median_time_ms(
    std::function<bool(const Config &, std::vector<RangeResult> &)> func_to_run,
    const Config &config, int samples, int iterations_per_sample) {
  if (samples <= 0 || iterations_per_sample <= 0)
    return -1.0;

  std::vector<double> iteration_times;
  iteration_times.reserve(samples * iterations_per_sample);
  std::vector<RangeResult> results_buffer;

  // Print scheduler info to terminal (not CSV)
  std::string scheduler_type;
  if (config.scheduling == SchedulingType::STATIC) {
    std::string variant;
    switch (config.static_variant) {
    case StaticVariant::BLOCK:
      variant = "Block";
      break;
    case StaticVariant::CYCLIC:
      variant = "Cyclic";
      break;
    case StaticVariant::BLOCK_CYCLIC:
      variant = "Block-Cyclic";
      break;
    }
    scheduler_type = "Static " + variant;
  } else if (config.scheduling == SchedulingType::DYNAMIC) {
    scheduler_type = "Dynamic";
  } else {
    scheduler_type = "Sequential";
  }

  std::cout << "  Running " << scheduler_type << " with " << config.num_threads
            << " thread(s)";
  if (scheduler_type != "Sequential") {
    std::cout << ", chunk size " << config.chunk_size;
  }
  std::cout << std::endl;

  for (int s = 0; s < samples; ++s) {
    // Only show sample number to terminal
    std::cout << "    Sample " << (s + 1) << "/" << samples << ": "
              << std::flush;
    int valid_iterations = 0;

    for (int iter = 0; iter < iterations_per_sample; ++iter) {
      Timer timer;
      bool success = func_to_run(config, results_buffer);
      double duration_ms = timer.elapsed_ms();

      if (!success) {
        std::cerr << "X" << std::flush;
        continue;
      }

      std::cout << "." << std::flush;
      valid_iterations++;
      iteration_times.push_back(duration_ms);
    }

    std::cout << std::endl;
  }

  if (iteration_times.empty())
    return -1.0;

  std::sort(iteration_times.begin(), iteration_times.end());
  size_t n = iteration_times.size();
  double median =
      (n % 2) ? iteration_times[n / 2]
              : (iteration_times[n / 2 - 1] + iteration_times[n / 2]) / 2.0;

  std::cout << "  → Median time: " << std::fixed << std::setprecision(4)
            << median << " ms" << std::endl;

  return median;
}

/**
 * @brief Creates and ensures a directory exists.
 */
bool ensure_directory_exists(const std::string &dir_path) {
  // This is a simplified version - in a real implementation,
  // you'd want to check if the directory exists and create it if not
  // using platform-specific code or a library
  system(("mkdir -p " + dir_path).c_str());
  return true;
}

/**
 * @brief Opens a CSV file for writing results.
 */
std::ofstream open_csv_file(const std::string &filename) {
  ensure_directory_exists(RESULTS_DIR);
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
  }
  return file;
}

/**
 * @brief Runs extended correctness tests.
 */
bool run_correctness_suite() {
  std::cout << "=== Running Extended Correctness Suite ===" << std::endl;
  int test_count = 0;
  int passed_count = 0;

  // Define test cases including edge cases
  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {1, 2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {1, 2, 4}, {1}},
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {1, 4, 8},
       {1, 10}},
      {"Larger Range", {{1, 10000}}, {1, 8, 16}, {64, 128}},
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {1, 4, 8}, {16}},
      // Edge cases
      {"Empty Range", {{50, 40}}, {1, 4}, {1}}, // Start > End
      {"Minimum Value", {{1, 1}}, {1, 4, 8}, {1, 16}},
      {"Boundary Case",
       {{4294967294, 4294967295}},
       {1, 4},
       {1, 64}},                                               // Near ULL max
      {"Large Chunk Size", {{1, 100}}, {1, 2, 4}, {200, 500}}, // Chunk > range
      {"More Threads Than Work",
       {{1, 10}},
       {16, 32},
       {1, 4}},                                 // Threads > range
      {"Zero Start", {{0, 10}}, {1, 4}, {1, 4}} // Should handle 0 correctly
  };

  for (const auto &tc : test_cases) {
    test_count++;
    bool testcase_success = true;
    std::cout << "\n[Test Case " << test_count << "]: " << tc.name << std::endl;

    std::cout << "  Executing Sequential baseline... " << std::flush;
    std::vector<ull> expected_results = run_sequential(tc.ranges);
    std::cout << "Done." << std::endl;

    for (int n_threads : tc.thread_counts) {
      for (ull chunk : tc.chunk_sizes) {
        // Test all static variants
        testcase_success &=
            test_static_variant(StaticVariant::BLOCK, n_threads, chunk,
                                tc.ranges, expected_results);

        testcase_success &=
            test_static_variant(StaticVariant::CYCLIC, n_threads, chunk,
                                tc.ranges, expected_results);

        testcase_success &=
            test_static_variant(StaticVariant::BLOCK_CYCLIC, n_threads, chunk,
                                tc.ranges, expected_results);

        // Test dynamic scheduler
        std::cout << "  [Dynamic T=" << n_threads << ", C=" << chunk
                  << "] Running..." << std::flush;
        std::vector<RangeResult> dynamic_results;

        Config config_dynamic;
        config_dynamic.scheduling = SchedulingType::DYNAMIC;
        config_dynamic.num_threads = n_threads;
        config_dynamic.chunk_size = chunk;
        config_dynamic.ranges = tc.ranges;
        config_dynamic.verbose = false;

        bool dynamic_success =
            run_dynamic_task_queue(config_dynamic, dynamic_results);
        if (dynamic_success &&
            compare_results(expected_results, dynamic_results, "Dynamic",
                            n_threads, chunk)) {
          std::cout << " PASS" << std::endl;
        } else {
          std::cout << " FAIL" << std::endl;
          testcase_success = false;
        }
      }
    }
    passed_count += (testcase_success ? 1 : 0);
  }

  std::cout << "\n=== Correctness Suite Summary ===" << std::endl;
  print_summary_line("Correctness", test_count, passed_count);
  std::cout << "===================================" << std::endl;
  return (test_count == passed_count);
}

/**
 * @brief Runs static scheduler comparison tests.
 */
bool run_static_performance_comparison(const std::vector<int> &thread_counts,
                                       const std::vector<ull> &chunk_sizes,
                                       int samples, int iterations_per_sample,
                                       const std::vector<Range> &workload) {
  std::cout << "\n=== Running Static Scheduler Comparison ===" << std::endl;

  // Open CSV file for writing results
  std::ofstream csv_file = open_csv_file(STATIC_COMPARISON_CSV);
  if (!csv_file.is_open()) {
    return false;
  }

  Config base_config;
  base_config.ranges = workload;
  base_config.verbose = false;
  base_config.scheduling = SchedulingType::STATIC;

  // Run sequential first as baseline
  Config seq_config = base_config;
  seq_config.num_threads = 1;
  seq_config.chunk_size = 1;
  seq_config.scheduling = SchedulingType::STATIC;

  std::function<bool(const Config &, std::vector<RangeResult> &)> seq_func =
      run_sequential_wrapper;

  std::cout << "Running sequential baseline..." << std::endl;
  double seq_time = measure_median_time_ms(seq_func, seq_config, samples,
                                           iterations_per_sample);

  if (seq_time > 0) {
    csv_file << "Sequential,1,N/A," << seq_time << std::endl;
  } else {
    csv_file << "Sequential,1,N/A,ERROR" << std::endl;
    return false;
  }

  // Lambda for running static scheduling with different variants
  auto run_static_with_variant = [](const Config &cfg,
                                    std::vector<RangeResult> &res) {
    return run_static_scheduling(cfg, res);
  };

  // Test each thread count and chunk size combination
  for (int n_threads : thread_counts) {
    for (ull chunk : chunk_sizes) {
      // Block variant
      Config config_block = base_config;
      config_block.static_variant = StaticVariant::BLOCK;
      config_block.num_threads = n_threads;
      config_block.chunk_size = chunk;

      std::cout << "\nTesting Block variant with " << n_threads
                << " threads, chunk size " << chunk << std::endl;
      double block_time =
          measure_median_time_ms(run_static_with_variant, config_block, samples,
                                 iterations_per_sample);

      if (block_time > 0) {
        double speedup = seq_time / block_time;
        csv_file << "Block," << n_threads << "," << chunk << "," << block_time
                 << "," << speedup << std::endl;
      } else {
        csv_file << "Block," << n_threads << "," << chunk << ",ERROR,0"
                 << std::endl;
      }

      // Cyclic variant
      Config config_cyclic = base_config;
      config_cyclic.static_variant = StaticVariant::CYCLIC;
      config_cyclic.num_threads = n_threads;
      config_cyclic.chunk_size = chunk;

      std::cout << "\nTesting Cyclic variant with " << n_threads
                << " threads, chunk size " << chunk << std::endl;
      double cyclic_time =
          measure_median_time_ms(run_static_with_variant, config_cyclic,
                                 samples, iterations_per_sample);

      if (cyclic_time > 0) {
        double speedup = seq_time / cyclic_time;
        csv_file << "Cyclic," << n_threads << "," << chunk << "," << cyclic_time
                 << "," << speedup << std::endl;
      } else {
        csv_file << "Cyclic," << n_threads << "," << chunk << ",ERROR,0"
                 << std::endl;
      }

      // Block-Cyclic variant
      Config config_block_cyclic = base_config;
      config_block_cyclic.static_variant = StaticVariant::BLOCK_CYCLIC;
      config_block_cyclic.num_threads = n_threads;
      config_block_cyclic.chunk_size = chunk;

      std::cout << "\nTesting Block-Cyclic variant with " << n_threads
                << " threads, chunk size " << chunk << std::endl;
      double block_cyclic_time =
          measure_median_time_ms(run_static_with_variant, config_block_cyclic,
                                 samples, iterations_per_sample);

      if (block_cyclic_time > 0) {
        double speedup = seq_time / block_cyclic_time;
        csv_file << "BlockCyclic," << n_threads << "," << chunk << ","
                 << block_cyclic_time << "," << speedup << std::endl;
      } else {
        csv_file << "BlockCyclic," << n_threads << "," << chunk << ",ERROR,0"
                 << std::endl;
      }
    }
  }

  csv_file.close();
  std::cout << "Static scheduler comparison results saved to "
            << STATIC_COMPARISON_CSV << std::endl;
  return true;
}

/**
 * @brief Runs performance tests for all scheduler types.
 */
bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload) {
  std::cout << "\n=== Running Performance Suite (All Schedulers) ==="
            << std::endl;

  // Open CSV file for writing results
  std::ofstream csv_file = open_csv_file(ALL_SCHEDULERS_CSV);
  if (!csv_file.is_open()) {
    return false;
  }

  Config base_config;
  base_config.ranges = workload;
  base_config.verbose = false;

  // Run sequential first as baseline
  Config seq_config = base_config;
  seq_config.num_threads = 1;
  seq_config.chunk_size = 1;

  std::function<bool(const Config &, std::vector<RangeResult> &)> seq_func =
      run_sequential_wrapper;

  std::cout << "Running sequential baseline..." << std::endl;
  double seq_time = measure_median_time_ms(seq_func, seq_config, samples,
                                           iterations_per_sample);

  if (seq_time > 0) {
    csv_file << "Sequential,1,N/A," << seq_time << std::endl;
  } else {
    csv_file << "Sequential,1,N/A,ERROR" << std::endl;
    return false;
  }

  // Function to run static block-cyclic scheduling (standard implementation)
  auto static_func = [](const Config &cfg, std::vector<RangeResult> &res) {
    return run_static_block_cyclic(cfg, res);
  };

  // Function to run dynamic scheduling
  auto dynamic_func = [](const Config &cfg, std::vector<RangeResult> &res) {
    return run_dynamic_task_queue(cfg, res);
  };

  // Test each thread count and chunk size combination
  for (int n_threads : thread_counts) {
    for (ull chunk : chunk_sizes) {
      // Static scheduling
      Config config_static = base_config;
      config_static.scheduling = SchedulingType::STATIC;
      config_static.num_threads = n_threads;
      config_static.chunk_size = chunk;

      std::cout << "\nTesting Static scheduler with " << n_threads
                << " threads, chunk size " << chunk << std::endl;
      double static_time = measure_median_time_ms(
          static_func, config_static, samples, iterations_per_sample);

      if (static_time > 0) {
        double speedup = seq_time / static_time;
        csv_file << "Static," << n_threads << "," << chunk << "," << static_time
                 << "," << speedup << std::endl;
      } else {
        csv_file << "Static," << n_threads << "," << chunk << ",ERROR,0"
                 << std::endl;
      }

      // Dynamic scheduling
      Config config_dynamic = base_config;
      config_dynamic.scheduling = SchedulingType::DYNAMIC;
      config_dynamic.num_threads = n_threads;
      config_dynamic.chunk_size = chunk;

      std::cout << "\nTesting Dynamic scheduler with " << n_threads
                << " threads, chunk size " << chunk << std::endl;
      double dynamic_time = measure_median_time_ms(
          dynamic_func, config_dynamic, samples, iterations_per_sample);

      if (dynamic_time > 0) {
        double speedup = seq_time / dynamic_time;
        csv_file << "Dynamic," << n_threads << "," << chunk << ","
                 << dynamic_time << "," << speedup << std::endl;
      } else {
        csv_file << "Dynamic," << n_threads << "," << chunk << ",ERROR,0"
                 << std::endl;
      }
    }
  }

  csv_file.close();
  std::cout << "Performance test results saved to " << ALL_SCHEDULERS_CSV
            << std::endl;
  return true;
}

/**
 * @brief Runs workload scaling tests.
 */
bool run_workload_scaling_tests(
    const std::vector<int> &thread_counts,
    const std::vector<std::vector<Range>> &workloads, int samples,
    int iterations_per_sample) {
  std::cout << "\n=== Running Workload Scaling Tests ===" << std::endl;

  // Open CSV file for writing results
  std::ofstream csv_file = open_csv_file(WORKLOAD_SCALING_CSV);
  if (!csv_file.is_open()) {
    return false;
  }

  // Run tests for each workload
  for (size_t workload_idx = 0; workload_idx < workloads.size();
       workload_idx++) {
    const auto &workload = workloads[workload_idx];

    // Describe the workload
    std::cout << "\nWorkload " << (workload_idx + 1) << ": ";
    for (size_t i = 0; i < workload.size(); ++i) {
      std::cout << "[" << workload[i].start << "-" << workload[i].end << "]";
      if (i < workload.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;

    Config base_config;
    base_config.ranges = workload;
    base_config.verbose = false;

    // Run sequential as baseline
    Config seq_config = base_config;
    seq_config.num_threads = 1;

    std::function<bool(const Config &, std::vector<RangeResult> &)> seq_func =
        run_sequential_wrapper;

    std::cout << "Running sequential baseline..." << std::endl;
    double seq_time = measure_median_time_ms(seq_func, seq_config, samples,
                                             iterations_per_sample);

    if (seq_time > 0) {
      csv_file << workload_idx << ",Sequential,1,N/A," << seq_time << std::endl;
    } else {
      csv_file << workload_idx << ",Sequential,1,N/A,ERROR" << std::endl;
      continue;
    }

    // Test different schedulers with fixed chunk size but varying thread counts
    const ull fixed_chunk_size = 64; // Fixed chunk size for this test

    // Functions for different scheduler types
    auto static_block_func = [](const Config &cfg,
                                std::vector<RangeResult> &res) {
      Config config_copy = cfg;
      config_copy.static_variant = StaticVariant::BLOCK;
      return run_static_scheduling(config_copy, res);
    };

    auto static_cyclic_func = [](const Config &cfg,
                                 std::vector<RangeResult> &res) {
      Config config_copy = cfg;
      config_copy.static_variant = StaticVariant::CYCLIC;
      return run_static_scheduling(config_copy, res);
    };

    auto static_block_cyclic_func = [](const Config &cfg,
                                       std::vector<RangeResult> &res) {
      Config config_copy = cfg;
      config_copy.static_variant = StaticVariant::BLOCK_CYCLIC;
      return run_static_scheduling(config_copy, res);
    };

    auto dynamic_func = [](const Config &cfg, std::vector<RangeResult> &res) {
      return run_dynamic_task_queue(cfg, res);
    };

    // Test each thread count
    for (int n_threads : thread_counts) {
      Config thread_config = base_config;
      thread_config.num_threads = n_threads;
      thread_config.chunk_size = fixed_chunk_size;
      thread_config.scheduling = SchedulingType::STATIC;

      // Static Block
      std::cout << "\nTesting Static Block with " << n_threads
                << " threads on workload " << (workload_idx + 1) << std::endl;
      double block_time = measure_median_time_ms(
          static_block_func, thread_config, samples, iterations_per_sample);

      if (block_time > 0) {
        double speedup = seq_time / block_time;
        csv_file << workload_idx << ",StaticBlock," << n_threads << ","
                 << fixed_chunk_size << "," << block_time << "," << speedup
                 << std::endl;
      } else {
        csv_file << workload_idx << ",StaticBlock," << n_threads << ","
                 << fixed_chunk_size << ",ERROR,0" << std::endl;
      }

      // Static Cyclic
      std::cout << "\nTesting Static Cyclic with " << n_threads
                << " threads on workload " << (workload_idx + 1) << std::endl;
      double cyclic_time = measure_median_time_ms(
          static_cyclic_func, thread_config, samples, iterations_per_sample);

      if (cyclic_time > 0) {
        double speedup = seq_time / cyclic_time;
        csv_file << workload_idx << ",StaticCyclic," << n_threads << ","
                 << fixed_chunk_size << "," << cyclic_time << "," << speedup
                 << std::endl;
      } else {
        csv_file << workload_idx << ",StaticCyclic," << n_threads << ","
                 << fixed_chunk_size << ",ERROR,0" << std::endl;
      }

      // Static Block-Cyclic
      std::cout << "\nTesting Static Block-Cyclic with " << n_threads
                << " threads on workload " << (workload_idx + 1) << std::endl;
      double block_cyclic_time =
          measure_median_time_ms(static_block_cyclic_func, thread_config,
                                 samples, iterations_per_sample);

      if (block_cyclic_time > 0) {
        double speedup = seq_time / block_cyclic_time;
        csv_file << workload_idx << ",StaticBlockCyclic," << n_threads << ","
                 << fixed_chunk_size << "," << block_cyclic_time << ","
                 << speedup << std::endl;
      } else {
        csv_file << workload_idx << ",StaticBlockCyclic," << n_threads << ","
                 << fixed_chunk_size << ",ERROR,0" << std::endl;
      }

      // Dynamic
      thread_config.scheduling = SchedulingType::DYNAMIC;
      std::cout << "\nTesting Dynamic with " << n_threads
                << " threads on workload " << (workload_idx + 1) << std::endl;
      double dynamic_time = measure_median_time_ms(
          dynamic_func, thread_config, samples, iterations_per_sample);

      if (dynamic_time > 0) {
        double speedup = seq_time / dynamic_time;
        csv_file << workload_idx << ",Dynamic," << n_threads << ","
                 << fixed_chunk_size << "," << dynamic_time << "," << speedup
                 << std::endl;
      } else {
        csv_file << workload_idx << ",Dynamic," << n_threads << ","
                 << fixed_chunk_size << ",ERROR,0" << std::endl;
      }
    }
  }

  csv_file.close();
  std::cout << "Workload scaling test results saved to " << WORKLOAD_SCALING_CSV
            << std::endl;
  return true;
}
