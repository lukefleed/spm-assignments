#include "testing.h"
#include "dynamic_scheduler.h"
#include "sequential.h"
#include "static_scheduler.h"
#include "utils.h" // Per Timer

#include <algorithm>  // std::sort, std::minmax_element
#include <cmath>      // std::sqrt, std::abs
#include <functional> // std::function
#include <iomanip>    // std::setw, std::fixed, std::setprecision
#include <iostream>   // std::clog
#include <numeric>    // std::accumulate
#include <vector>

// --- Helper Functions ---

// Compares expected sequential results with the results returned by a scheduler
// test. Returns true if they match; otherwise, prints detailed mismatch
// information.
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

// Prints a summary line in a formatted way
void print_summary_line(const std::string &testName, int total, int passed) {
  std::cout << std::setw(20) << std::left << testName
            << " Total: " << std::setw(4) << total
            << " Passed: " << std::setw(4) << passed
            << " Failed: " << std::setw(4) << (total - passed) << std::endl;
}

// --- Correctness Suite ---

struct CorrectnessTestCase {
  std::string name;
  std::vector<Range> ranges;
  std::vector<int> thread_counts;
  std::vector<ull> chunk_sizes;
};

bool run_correctness_suite() {
  std::cout << "=== Running Correctness Suite ===" << std::endl;
  int test_count = 0;
  int passed_count = 0;

  std::vector<CorrectnessTestCase> test_cases = {
      {"Small Range", {{1, 100}}, {1, 2, 4}, {1, 8, 32}},
      {"Single Value Range", {{27, 27}}, {1, 2}, {1}},
      {"Multiple Small Ranges",
       {{1, 10}, {50, 60}, {100, 110}},
       {4, 8},
       {1, 10}},
      {"Larger Range", {{1, 10000}}, {8, 16}, {64, 128}},
      {"Mixed Ranges", {{10, 20}, {1000, 1500}, {80, 90}}, {4}, {16}}};

  for (const auto &tc : test_cases) {
    test_count++;
    bool testcase_success = true;
    std::cout << "\n[Test Case " << test_count << "]: " << tc.name << std::endl;

    // Run the sequential implementation (baseline)
    std::cout << "  Executing Sequential baseline... " << std::flush;
    std::vector<ull> expected_results = run_sequential(tc.ranges);
    std::cout << "Done." << std::endl;

    // Run tests for static and dynamic schedulers
    for (int n_threads : tc.thread_counts) {
      for (ull chunk : tc.chunk_sizes) {
        // Config for static scheduling
        Config config_static;
        config_static.scheduling = SchedulingType::STATIC;
        config_static.num_threads = n_threads;
        config_static.chunk_size = chunk;
        config_static.ranges = tc.ranges;
        config_static.verbose = false;

        // Config for dynamic scheduling (copy of static)
        Config config_dynamic = config_static;
        config_dynamic.scheduling = SchedulingType::DYNAMIC;

        // --- Static Test ---
        std::cout << "  [Static  T=" << n_threads << ", C=" << chunk
                  << "] Running..." << std::flush;
        std::vector<RangeResult> static_results;
        bool static_success =
            run_static_block_cyclic(config_static, static_results);
        if (static_success && compare_results(expected_results, static_results,
                                              "Static", n_threads, chunk)) {
          std::cout << " PASS" << std::endl;
        } else {
          std::cout << " FAIL" << std::endl;
          testcase_success = false;
        }

        // --- Dynamic Test ---
        std::cout << "  [Dynamic T=" << n_threads << ", C=" << chunk
                  << "] Running..." << std::flush;
        std::vector<RangeResult> dynamic_results;
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

// --- Performance Suite ---

// Wrapper for sequential implementation to match the function signature needed
// for measurement
bool run_sequential_wrapper(const Config &cfg, std::vector<RangeResult> &res) {
  std::vector<ull> seq_results = run_sequential(cfg.ranges);
  res.clear();
  res.reserve(seq_results.size());
  for (ull result : seq_results) {
    RangeResult rr;
    rr.max_steps.store(result);
    res.push_back(rr);
  }
  return true;
}

// Helper function to measure median execution time in ms.
double measure_median_time_ms(
    std::function<bool(const Config &, std::vector<RangeResult> &)> func_to_run,
    const Config &config, int samples, int iterations_per_sample) {
  if (samples <= 0 || iterations_per_sample <= 0)
    return -1.0;

  std::vector<double> iteration_times;
  iteration_times.reserve(samples * iterations_per_sample);
  std::vector<RangeResult> results_buffer;

  std::cout << "  Running measurements for "
            << (config.scheduling == SchedulingType::STATIC
                    ? "Static"
                    : (config.scheduling == SchedulingType::DYNAMIC
                           ? "Dynamic"
                           : "Sequential"))
            << " scheduler with " << config.num_threads << " thread(s)"
            << (config.scheduling == SchedulingType::STATIC ||
                        config.scheduling == SchedulingType::DYNAMIC
                    ? " and chunk size " + std::to_string(config.chunk_size)
                    : "")
            << std::endl;

  for (int s = 0; s < samples; ++s) {
    std::cout << "    Sample " << (s + 1) << "/" << samples << ": "
              << std::flush;
    double sample_total = 0.0;
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
      sample_total += duration_ms;
      valid_iterations++;
      iteration_times.push_back(duration_ms);
    }

    if (valid_iterations > 0) {
      double avg = sample_total / valid_iterations;
      std::cout << " Avg: " << std::fixed << std::setprecision(2) << avg
                << " ms" << std::endl;
    } else {
      std::cout << " Failed" << std::endl;
    }
  }

  if (iteration_times.empty())
    return -2.0;

  std::sort(iteration_times.begin(), iteration_times.end());
  size_t n = iteration_times.size();
  double median =
      (n % 2) ? iteration_times[n / 2]
              : (iteration_times[n / 2 - 1] + iteration_times[n / 2]) / 2.0;

  std::cout << "  → Median execution time: " << std::fixed
            << std::setprecision(4) << median << " ms over "
            << iteration_times.size() << " measurements" << std::endl;

  return median;
}

bool run_performance_suite(const std::vector<int> &thread_counts,
                           const std::vector<ull> &chunk_sizes, int samples,
                           int iterations_per_sample,
                           const std::vector<Range> &workload) {
  std::cout << "\n=== Running Performance Suite ===" << std::endl;
  std::cout << "Samples/Config: " << samples
            << ", Iterations/Sample: " << iterations_per_sample << std::endl;

  std::cout << "Workload Ranges: ";
  for (size_t i = 0; i < workload.size(); ++i) {
    const auto &r = workload[i];
    std::cout << "[" << r.start << "-" << r.end << "]";
    if (i < workload.size() - 1)
      std::cout << ", ";
  }
  std::cout << std::endl << std::endl;
  std::cout << "Scheduler,Threads,ChunkSize,MedianTimeMs" << std::endl;

  Config base_config;
  base_config.ranges = workload;
  base_config.verbose = false;

  // --- Sequential Baseline ---
  Config seq_config = base_config;
  seq_config.num_threads = 1;
  seq_config.chunk_size = 0;
  seq_config.scheduling = SchedulingType::STATIC;
  std::function<bool(const Config &, std::vector<RangeResult> &)> seq_func =
      [](const Config &cfg, std::vector<RangeResult> &res) {
        return run_sequential_wrapper(cfg, res);
      };
  double median_seq_ms = measure_median_time_ms(seq_func, seq_config, samples,
                                                iterations_per_sample);
  if (median_seq_ms >= 0) {
    std::cout << "Sequential,1,N/A," << std::fixed << std::setprecision(4)
              << median_seq_ms << std::endl;
  } else {
    std::cout << "Sequential,1,N/A,ERROR" << std::endl;
  }

  // --- Static and Dynamic Tests ---
  for (int n_threads : thread_counts) {
    for (ull chunk : chunk_sizes) {
      // Static
      Config config_static = base_config;
      config_static.scheduling = SchedulingType::STATIC;
      config_static.num_threads = n_threads;
      config_static.chunk_size = chunk;
      auto static_func = [](const Config &cfg, std::vector<RangeResult> &res) {
        return run_static_block_cyclic(cfg, res);
      };
      double median_static_ms = measure_median_time_ms(
          static_func, config_static, samples, iterations_per_sample);
      std::cout << "Static," << n_threads << "," << chunk << ",";
      if (median_static_ms >= 0) {
        std::cout << std::fixed << std::setprecision(4) << median_static_ms;
      } else {
        std::cout << "ERROR";
      }
      std::cout << std::endl;

      // Dynamic
      Config config_dynamic = base_config;
      config_dynamic.scheduling = SchedulingType::DYNAMIC;
      config_dynamic.num_threads = n_threads;
      config_dynamic.chunk_size = chunk;
      auto dynamic_func = [](const Config &cfg, std::vector<RangeResult> &res) {
        return run_dynamic_task_queue(cfg, res);
      };
      double median_dynamic_ms = measure_median_time_ms(
          dynamic_func, config_dynamic, samples, iterations_per_sample);
      std::cout << "Dynamic," << n_threads << "," << chunk << ",";
      if (median_dynamic_ms >= 0) {
        std::cout << std::fixed << std::setprecision(4) << median_dynamic_ms;
      } else {
        std::cout << "ERROR";
      }
      std::cout << std::endl;
    }
  }
  return true;
}
