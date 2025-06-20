/**
 * @file bench_main.cpp
 * @brief Benchmark driver for minizp performance evaluation.
 *
 * Sets up test data, sweeps over thread counts and block sizes,
 * measures sequential and parallel throughput, and outputs results
 * to CSV and stdout.
 */

#include "bench_utils.hpp"
#include "compressor.hpp"
#include "config.hpp"
#include "file_handler.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <cmath> // For std::min, std::max, std::sqrt
#include <cstdlib>
#include <filesystem>
#include <fstream> // For CSV output
#include <iomanip> // For formatting output
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// --- Benchmark Configuration ---
const std::string BENCH_DIR =
    "./test_data_bench_cpp"; ///< Directory for benchmark data.

/**
 * @brief Holds parameters for benchmark runs, including thread sweep and file
 * sizes.
 */
struct BenchParams {
  std::string type =
      "one_large"; ///< Type of benchmark: one_large or many_small
  int threads = omp_get_max_threads(); ///< Number of threads to test
  int iterations = 2;                  ///< Number of benchmark iterations
  int warmup = 1;                      ///< Number of warmup runs
  size_t large_file_size =
      512 * 1024 * 1024;      ///< Size for one large file (bytes)
  int num_small_files = 4000; ///< Number of small files to generate
  size_t min_small_file_size =
      1 * 1024; ///< Minimum size for small files (bytes)
  size_t max_small_file_size =
      1 * 1024 * 1024; ///< Maximum size for small files (bytes)
  ConfigData config;   ///< Compression configuration
  std::vector<size_t> block_sizes_list; ///< Block sizes for matrix sweep
};

/**
 * @brief Parses command-line arguments for benchmark configuration.
 *
 * Expects key=value pairs prefixed with '--'. Validates types and ranges.
 *
 * @param argc Argument count from main.
 * @param argv Argument values from main.
 * @param[out] params Structure to populate with parsed values.
 * @return true if parsing succeeded, false on error.
 */
bool parseBenchArgs(int argc, char *argv[], BenchParams &params) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    size_t eq_pos = arg.find('=');
    if (arg.rfind("--", 0) == 0 && eq_pos != std::string::npos) {
      args[arg.substr(2, eq_pos - 2)] = arg.substr(eq_pos + 1);
    }
  }
  try {
    if (args.count("type"))
      params.type = args["type"];
    if (args.count("threads"))
      params.threads = std::stoi(args["threads"]);
    if (args.count("iterations"))
      params.iterations = std::stoi(args["iterations"]);
    if (args.count("warmup"))
      params.warmup = std::stoi(args["warmup"]);
    if (args.count("large_size"))
      params.large_file_size = std::stoull(args["large_size"]);
    if (args.count("num_small"))
      params.num_small_files = std::stoi(args["num_small"]);
    if (args.count("verbosity"))
      params.config.verbosity = std::stoi(args["verbosity"]);
    if (args.count("threshold"))
      params.config.large_file_threshold = std::stoull(args["threshold"]);
    if (args.count("blocksize"))
      params.config.block_size = std::stoull(args["blocksize"]);
    if (args.count("block_sizes_list")) {
      std::string list = args["block_sizes_list"];
      size_t pos = 0;
      while ((pos = list.find(',')) != std::string::npos) {
        params.block_sizes_list.push_back(std::stoull(list.substr(0, pos)));
        list.erase(0, pos + 1);
      }
      if (!list.empty())
        params.block_sizes_list.push_back(std::stoull(list));
    }
    if (args.count("min_size"))
      params.min_small_file_size = std::stoull(args["min_size"]);
    if (args.count("max_size"))
      params.max_small_file_size = std::stoull(args["max_size"]);
    // Validations
    if (params.type != "one_large" && params.type != "many_small" &&
        params.type != "many_large_sequential" &&
        params.type != "many_large_parallel" &&
        params.type != "many_large_parallel_right")
      throw std::runtime_error("Invalid type");
    if (params.threads <= 0)
      throw std::runtime_error("Threads must be positive");
    if (params.min_small_file_size > params.max_small_file_size)
      throw std::runtime_error("min_size must not exceed max_size");
  } catch (const std::exception &e) {
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief Prepares the benchmark environment by creating or cleaning the data
 * directory.
 *
 * Generates either one large file or many small files of random sizes
 * based on the 'type' field in params.
 *
 * @param params Benchmark parameters controlling data generation.
 * @throws runtime_error on file system or generation failure.
 */
void setup_bench_environment(const BenchParams &params) {
  std::cout << "Setting up benchmark environment in " << BENCH_DIR << "..."
            << std::endl;
  std::error_code ec;
  fs::remove_all(BENCH_DIR, ec);
  fs::create_directories(BENCH_DIR, ec);
  if (ec)
    throw std::runtime_error("Failed dir creation: " + ec.message());

  if (params.type == "one_large") {
    const std::string file_path = std::string(BENCH_DIR) + "/large_file.bin";
    if (!TestUtils::create_random_file(file_path, params.large_file_size,
                                       0)) { // Verbosity 0 for bench setup
      throw std::runtime_error("Failed creation: large file");
    }
  } else if (params.type == "many_large_sequential" ||
             params.type == "many_large_parallel" ||
             params.type == "many_large_parallel_right") {
    // Generate many large files
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(50ULL * 1024 * 1024,
                                               250ULL * 1024 * 1024);
    for (int i = 0; i < 10; ++i) {
      size_t size = dist(gen);
      std::string file_path =
          std::string(BENCH_DIR) + "/large_file_" + std::to_string(i) + ".bin";
      if (!TestUtils::create_random_file(file_path, size, 0)) {
        throw std::runtime_error("Failed creation: large file " +
                                 std::to_string(i));
      }
    }
  } else { // many_small
    // Generate many small files with random, distinct sizes in user-defined
    // range
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(params.min_small_file_size,
                                               params.max_small_file_size);
    std::unordered_set<size_t> used_sizes;
    for (int i = 0; i < params.num_small_files; ++i) {
      size_t size;
      do {
        size = dist(gen);
      } while (!used_sizes.insert(size).second);
      const std::string file_path =
          std::string(BENCH_DIR) + "/small_file_" + std::to_string(i) + ".bin";
      if (!TestUtils::create_random_file(file_path, size, 0)) {
        throw std::runtime_error("Failed creation: small file " +
                                 std::to_string(i));
      }
    }
  }
  std::cout << "Setup complete." << std::endl;
}

/**
 * @brief Cleans up the benchmark environment by removing generated files.
 */
void cleanup_bench_environment() {
  std::cout << "Cleaning up benchmark environment..." << std::endl;
  std::error_code ec;
  fs::remove_all(BENCH_DIR, ec);
}

/**
 * @brief Executes the compression workflow over a list of work items.
 *
 * Uses OpenMP parallel for to process files and sets an error flag
 * on the first failure. The number of threads used for the outer loop
 * is determined by the `omp_set_num_threads` call before invoking the
 * benchmark lambda containing this function. The number of threads used
 * for inner parallelism (within `Compressor::process_file`) is determined
 * by `cfg.num_threads`.
 *
 * @param items List of file work items discovered for processing.
 * @param cfg Compression configuration including the number of threads for
 *            inner parallelism.
 * @return true if all files processed successfully, false otherwise.
 */
bool perform_compression_work(const std::vector<FileHandler::WorkItem> &items,
                              ConfigData &cfg) {
  std::atomic<bool> error_flag = false;
  // This outer loop parallelizes over the files.
  // The number of threads is controlled by omp_set_num_threads() before the
  // call.
#pragma omp parallel for default(none) shared(items, cfg, error_flag)          \
    schedule(dynamic)
  for (size_t i = 0; i < items.size(); ++i) {
    if (error_flag.load())
      continue;
    // Compressor::process_file uses cfg.num_threads for inner block
    // parallelism.
    if (!Compressor::process_file(items[i].path, cfg))
      error_flag = true;
  }
  return !error_flag.load();
}

/**
 * @brief Main entry point for the benchmark application.
 *
 * Parses parameters, initializes environment, discovers work items,
 * sweeps sequential and parallel runs (or block-size matrix), outputs CSV,
 * and cleans up.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of argument strings.
 * @return 0 on successful benchmark, non-zero on error.
 */
int main(int argc, char *argv[]) {
  BenchParams params;
  if (!parseBenchArgs(argc, argv, params))
    return 1;

  params.config.verbosity = 0;
  params.config.compress_mode = true;
  params.config.remove_origin = false;
  params.config.recurse = true; // we want to find all generated files

  // Configure OpenMP settings globally where applicable
  omp_set_max_active_levels(2); // Allow up to 2 levels of nesting

  std::cout << "--- Benchmark Initializing ---" << std::endl;
  std::cout << "Type: " << params.type << ", Threads: " << params.threads
            << ", Iterations: " << params.iterations
            << ", Warmup: " << params.warmup;
  if (params.type == "one_large") {
    std::cout << ", File Size: " << params.large_file_size << " bytes";
  } else if (params.type == "many_small") {
    std::cout << ", Num Small Files: " << params.num_small_files << ", Sizes: ["
              << (params.min_small_file_size / 1024) << " - "
              << (params.max_small_file_size / (1024)) << "] KiB";
  } else if (params.type == "many_large_sequential" ||
             params.type == "many_large_parallel" ||
             params.type == "many_large_parallel_right") {
    static const size_t MIN_LARGE = 50ULL * 1024 * 1024;
    static const size_t MAX_LARGE = 250ULL * 1024 * 1024;
    std::cout << ", Num Large Files: 10"
              << ", Size Range: [" << (MIN_LARGE / (1024 * 1024)) << " - "
              << (MAX_LARGE / (1024 * 1024)) << "] MiB";
  }
  std::cout << std::endl;

  try {
    setup_bench_environment(params);
    std::vector<std::string> initial_paths = {BENCH_DIR};
    auto work_items =
        FileHandler::discover_work_items(initial_paths, params.config);
    if (work_items.empty())
      throw std::runtime_error("No work items found.");
    const int num_work_items = static_cast<int>(work_items.size());

    // --- Automated Benchmark Sweep ---
    if (params.type == "many_small") {
      /**
       * @brief Benchmark for many small files: sweep threads only.
       * Parallelism is only applied over the files.
       */
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      std::ofstream csv_small("results/data/benchmark_many_small.csv");
      csv_small << "threads,seq_time_s,par_time_s,speedup" << '\n';

      // Sequential baseline (1 thread for outer loop)
      ConfigData cfg_seq = params.config;
      cfg_seq.num_threads = 1; // Inner parallelism is irrelevant here
      omp_set_num_threads(1);
      omp_set_nested(false); // No nesting needed
      auto seq_work_small = [&]() {
        TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
        return perform_compression_work(work_items, cfg_seq);
      };
      BenchUtils::BenchmarkResult seq_result = BenchUtils::run_benchmark(
          seq_work_small, params.iterations, params.warmup);
      if (!seq_result.success) {
        throw std::runtime_error(
            "Sequential baseline benchmark failed with code: " +
            std::to_string(static_cast<int>(seq_result.error_code)));
      }
      double time_seq = seq_result.median_time_s;

      // Print and record baseline
      std::cout << "ManySmall: Seq baseline=" << std::fixed
                << std::setprecision(2) << time_seq << "s" << std::endl;
      csv_small << 1 << "," << time_seq << "," << time_seq << ",1" << '\n';

      // Parallel runs
      std::cout << std::setw(10) << "Threads" << std::setw(12) << "Par(s)"
                << std::setw(10) << "Speedup" << std::endl;
      for (int th : thread_counts) {
        if (th == 1)
          continue;
        ConfigData cfg_p = params.config;
        cfg_p.num_threads = 1; // Inner parallelism irrelevant
        omp_set_num_threads(
            th); // Set threads for the outer loop in perform_compression_work
        omp_set_nested(false); // No nesting needed
        auto par_work_small = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          return perform_compression_work(work_items, cfg_p);
        };
        BenchUtils::BenchmarkResult par_result = BenchUtils::run_benchmark(
            par_work_small, params.iterations, params.warmup);
        if (!par_result.success) {
          std::cerr << "Warning: Parallel benchmark failed for threads=" << th
                    << " with code: " << static_cast<int>(par_result.error_code)
                    << std::endl;
          continue; // Skip this data point
        }
        double time_par = par_result.median_time_s;
        double speed = time_seq / time_par;
        csv_small << th << "," << time_seq << "," << time_par << "," << speed
                  << '\n';
        std::cout << std::setw(10) << th << std::setw(12) << std::fixed
                  << std::setprecision(2) << time_par << std::setw(10)
                  << std::fixed << std::setprecision(2) << speed << std::endl;
      }
      csv_small.close();
      cleanup_bench_environment();
      return 0;
    }

    if (params.type == "many_large_sequential") {
      /**
       * @brief Matrix benchmark: Sequential file dispatch, nested block
       * parallelism. Sweeps over block sizes and inner thread counts. The outer
       * loop over files is always sequential C++.
       */
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);

      std::vector<size_t> block_sizes;
      if (!params.block_sizes_list.empty()) {
        block_sizes = params.block_sizes_list;
      } else {
        for (int i = 1; i <= 12; ++i)
          block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);
      }

      std::ofstream csv_seq("results/data/benchmark_many_large_sequential.csv");
      csv_seq << "block_size,threads,seq_time_s,par_time_s,speedup" << '\n';

      // Outer loop over block sizes
      for (size_t bs : block_sizes) {
        params.config.block_size = bs; // Set current block size

        // Baseline: single-thread inner parallelism for this block size
        ConfigData cfg_b = params.config;
        double base_time;
        {
          cfg_b.num_threads = 1;  // Inner parallelism uses 1 thread
          omp_set_nested(true);   // Enable nesting (though inner is 1 thread)
          omp_set_num_threads(1); // Outer loop is sequential C++
          auto work = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            // Sequential C++ loop over files
            for (auto &item : work_items) {
              // process_file uses cfg_b.num_threads (1) for inner parallelism
              if (!Compressor::process_file(item.path, cfg_b))
                return false;
            }
            return true;
          };
          auto r =
              BenchUtils::run_benchmark(work, params.iterations, params.warmup);
          if (!r.success)
            throw std::runtime_error(
                "many_large_sequential baseline failed for block_size=" +
                std::to_string(bs));
          base_time = r.median_time_s;
        }
        // Record baseline for this block size
        csv_seq << bs << "," << 1 << "," << base_time << "," << base_time
                << ",1" << '\n';

        // Print section header and table header for this block size
        std::cout << "\nBlockSize=" << (bs / (1024 * 1024)) << " MiB"
                  << "  Seq(s)=" << std::fixed << std::setprecision(2)
                  << base_time << "s (Sequential Dispatch)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::endl;

        // Inner loop over inner thread counts for this block size
        for (int th : thread_counts) {
          if (th == 1)
            continue; // Skip baseline, already recorded

          ConfigData cfg_p = params.config; // Use current block size
          cfg_p.num_threads = th; // Inner parallelism uses 'th' threads
          omp_set_nested(true);   // Allow inner parallelism
          // Set OMP threads for inner calls within process_file
          omp_set_num_threads(th);

          auto work_p = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            // Sequential C++ loop over files
            for (auto &item : work_items) {
              // process_file uses cfg_p.num_threads (th) for inner parallelism
              if (!Compressor::process_file(item.path, cfg_p))
                return false;
            }
            return true;
          };
          auto rp = BenchUtils::run_benchmark(work_p, params.iterations,
                                              params.warmup);
          if (!rp.success) {
            std::cerr << "Warning: many_large_sequential benchmark failed for "
                         "threads="
                      << th << " block_size=" << bs << std::endl;
            continue;
          }
          double tp = rp.median_time_s;
          double sp = base_time / tp;
          // Record result for this block size and thread count
          csv_seq << bs << "," << th << "," << base_time << "," << tp << ","
                  << sp << '\n';
          std::cout << std::setw(8) << th << std::setw(12) << std::fixed
                    << std::setprecision(2) << tp << std::setw(10) << std::fixed
                    << std::setprecision(2) << sp << std::endl;
        }
      } // End loop over block sizes
      csv_seq.close();
      cleanup_bench_environment();
      return 0;
    }

    if (params.type == "many_large_parallel") {
      /**
       * @brief Matrix benchmark: Oversubscribed nested parallelism.
       * Sweeps over block sizes and thread counts 'p'. Both the outer file loop
       * and the inner block loop use 'p' threads, potentially leading to p*p
       * threads.
       */
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);

      std::vector<size_t> block_sizes;
      if (!params.block_sizes_list.empty()) {
        block_sizes = params.block_sizes_list;
      } else {
        for (int i = 1; i <= 12; ++i)
          block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);
      }

      std::ofstream csv("results/data/benchmark_many_large_parallel.csv");
      csv << "block_size,threads,seq_time_s,par_time_s,speedup" << '\n';

      // Loop over block sizes
      for (size_t bs : block_sizes) {
        params.config.block_size = bs;

        // Sequential baseline: 1 thread total, no nesting
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1; // Inner uses 1
        omp_set_nested(false);
        omp_set_num_threads(1); // Outer uses 1
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          return perform_compression_work(work_items, cfg_seq);
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        if (!res_bs.success)
          throw std::runtime_error("many_large_parallel baseline failed");
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq << ",1"
            << '\n';

        // Print section header and table header
        std::cout << "\nBlockSize=" << (bs / (1024 * 1024)) << " MiB"
                  << "  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s (Oversubscribed Nesting)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::endl;

        // Parallel runs per thread count > 1
        for (int th : thread_counts) {
          if (th == 1)
            continue;
          ConfigData cfg_p = params.config;
          cfg_p.num_threads = th;  // Inner loop uses 'th' threads
          omp_set_nested(true);    // Enable nesting
          omp_set_num_threads(th); // Outer loop also uses 'th' threads
          auto par_work = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            return perform_compression_work(work_items, cfg_p);
          };
          auto pres = BenchUtils::run_benchmark(par_work, params.iterations,
                                                params.warmup);
          if (!pres.success) {
            std::cerr
                << "Warning: many_large_parallel benchmark failed for threads="
                << th << " block_size=" << bs << std::endl;
            continue;
          }
          double time_par = pres.median_time_s;
          double speed = time_seq / time_par;
          csv << bs << "," << th << "," << time_seq << "," << time_par << ","
              << speed << '\n';
          std::cout << std::setw(8) << th << std::setw(12) << std::fixed
                    << std::setprecision(2) << time_par << std::setw(10)
                    << std::fixed << std::setprecision(2) << speed << std::endl;
        }
      }
      csv.close();
      cleanup_bench_environment();
      return 0;
    }

    if (params.type == "many_large_parallel_right") {
      /**
       * @brief Matrix benchmark: Controlled nested parallelism.
       * Sweeps over block sizes and total thread counts 'p'. 'p' is distributed
       * between the outer file loop (t_outer) and inner block loop (t_inner)
       * aiming for t_outer * t_inner <= p.
       */
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);

      std::vector<size_t> block_sizes;
      if (!params.block_sizes_list.empty()) {
        block_sizes = params.block_sizes_list;
      } else {
        for (int i = 1; i <= 12; ++i)
          block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);
      }

      std::ofstream csv("results/data/benchmark_many_large_parallel_right.csv");
      csv << "block_size,threads,seq_time_s,par_time_s,speedup,t_outer,t_inner"
          << '\n';

      for (size_t bs : block_sizes) {
        params.config.block_size = bs;

        // Sequential baseline: 1 thread total, no nesting.
        // Use sequential C++ loop for consistency with many_large_sequential
        // baseline.
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1;
        omp_set_nested(false);
        omp_set_num_threads(1);
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          for (auto &item : work_items) {
            if (!Compressor::process_file(item.path, cfg_seq))
              return false;
          }
          return true;
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        if (!res_bs.success)
          throw std::runtime_error("many_large_parallel_right baseline failed");
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq
            << ",1,1,1" // Baseline uses 1x1
            << '\n';

        std::cout << "\nBlockSize=" << (bs / (1024 * 1024)) << " MiB"
                  << "  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s (Controlled Nesting)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::setw(8) << "T_out"
                  << std::setw(8) << "T_in" << std::endl;

        // Parallel runs with controlled nesting
        for (int p : thread_counts) {
          if (p == 1)
            continue;

          // Calculate thread distribution
          int ideal_outer =
              static_cast<int>(std::ceil(std::sqrt(static_cast<double>(p))));
          int t_outer =
              std::min(ideal_outer, num_work_items); // Limit by number of files
          t_outer =
              std::min(t_outer, p); // Cannot use more threads than requested
          t_outer = std::max(1, t_outer); // Ensure at least 1 outer thread

          // Calculate t_inner based on remaining budget, ensure at least 1
          int t_inner = (t_outer > 0) ? std::max(1, p / t_outer) : 1;

          ConfigData cfg_p_right = params.config;
          cfg_p_right.num_threads = t_inner; // Pass t_inner to inner loops

          omp_set_nested(true);
          omp_set_max_active_levels(2);
          // Set threads for the outer parallel region in
          // perform_compression_work
          omp_set_num_threads(t_outer);

          auto par_work_right = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            // perform_compression_work creates t_outer threads.
            // Each thread calling process_file uses cfg_p_right.num_threads
            // (t_inner).
            return perform_compression_work(work_items, cfg_p_right);
          };

          auto pres = BenchUtils::run_benchmark(
              par_work_right, params.iterations, params.warmup);
          if (!pres.success) {
            std::cerr << "Warning: many_large_parallel_right benchmark failed "
                         "for threads="
                      << p << " block_size=" << bs << std::endl;
            continue;
          }
          double time_par = pres.median_time_s;
          double speed = time_seq / time_par;
          csv << bs << "," << p << "," << time_seq << "," << time_par << ","
              << speed << "," << t_outer << "," << t_inner << '\n';

          std::cout << std::setw(8) << p << std::setw(12) << std::fixed
                    << std::setprecision(2) << time_par << std::setw(10)
                    << std::fixed << std::setprecision(2) << speed
                    << std::setw(8) << t_outer << std::setw(8) << t_inner
                    << std::endl;
        }
      }
      csv.close();
      cleanup_bench_environment();
      return 0;
    }

    if (params.type == "one_large") {
      /**
       * @brief Matrix benchmark for a single large file.
       * Sweeps over block sizes and thread counts 'p'. Parallelism is only
       * applied within the file (block parallelism). Nesting is disabled.
       */
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);

      std::vector<size_t> block_sizes;
      if (!params.block_sizes_list.empty()) {
        block_sizes = params.block_sizes_list;
      } else {
        for (int i = 1; i <= 12; ++i)
          block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);
      }

      std::ofstream csv("results/data/benchmark_one_large.csv");
      csv << "block_size,threads,seq_time_s,par_time_s,speedup" << '\n';

      for (size_t bs : block_sizes) {
        params.config.block_size = bs; // Set current block size

        // Sequential baseline for this block size (1 thread for block
        // parallelism)
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1;
        omp_set_num_threads(1); // Set threads for the single parallel region
                                // inside process_file
        omp_set_nested(false);  // No nesting needed
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          // perform_compression_work has only one item, inner parallelism uses
          // 1 thread
          return perform_compression_work(work_items, cfg_seq);
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        if (!res_bs.success)
          throw std::runtime_error("one_large baseline failed");
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq << ",1"
            << '\n';

        std::cout << "\nBlockSize=" << (bs / (1024 * 1024)) << " MiB"
                  << "  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s (One Large)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::endl;

        // Parallel runs for this block size
        for (int th : thread_counts) {
          if (th == 1)
            continue;

          ConfigData cfg_par = params.config; // Use current block size
          cfg_par.num_threads =
              th; // Set thread count for inner block parallelism
          omp_set_num_threads(
              th); // Set threads for the parallel region inside process_file
          // set to true
          omp_set_nested(true);

          auto par_work = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            // perform_compression_work uses cfg_par.num_threads for block
            // parallelism
            return perform_compression_work(work_items, cfg_par);
          };
          auto pres = BenchUtils::run_benchmark(par_work, params.iterations,
                                                params.warmup);
          if (!pres.success) {
            std::cerr
                << "Warning: one_large parallel benchmark failed for threads="
                << th << " block_size=" << bs << std::endl;
            continue;
          }
          double time_par = pres.median_time_s;
          double speed = time_seq / time_par;
          csv << bs << "," << th << "," << time_seq << "," << time_par << ","
              << speed << '\n';
          std::cout << std::setw(8) << th << std::setw(12) << std::fixed
                    << std::setprecision(2) << time_par << std::setw(10)
                    << std::fixed << std::setprecision(2) << speed << std::endl;
        }
      }
      csv.close();
      cleanup_bench_environment();
      return 0;
    }

    // Should not reach here if type matched one of the branches
    throw std::runtime_error("Benchmark type not handled: " + params.type);

  } catch (const std::exception &e) {
    std::cerr << "\n!!! Benchmark FAILED: " << e.what() << std::endl;
    cleanup_bench_environment();
    return 1;
  }
}
