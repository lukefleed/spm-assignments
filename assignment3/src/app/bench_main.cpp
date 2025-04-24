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
#include <vector> // Ensure vector is included

namespace fs = std::filesystem;

// --- Benchmark Configuration ---
const std::string BENCH_DIR = "./test_data_bench_cpp";

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
        params.type != "many_large_parallel_right") // Add new type
      throw std::runtime_error("Invalid type");
    if (params.threads <= 0)
      throw std::runtime_error("Threads must be positive");
    if (params.min_small_file_size > params.max_small_file_size)
      throw std::runtime_error("min_size must not exceed max_size");
    // ... other validations
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
             params.type == "many_large_parallel_right") { // Add new type
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
 * on the first failure.
 *
 * @param items List of file work items discovered for processing.
 * @param cfg Compression configuration including thread count.
 * @return true if all files processed successfully, false otherwise.
 */
bool perform_compression_work(const std::vector<FileHandler::WorkItem> &items,
                              ConfigData &cfg) {
  std::atomic<bool> error_flag = false;
#pragma omp parallel for if (cfg.num_threads > 1) default(none)                \
    shared(items, cfg, error_flag) schedule(dynamic)
  for (size_t i = 0; i < items.size(); ++i) {
    if (error_flag.load())
      continue;
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

  // Ensure low verbosity for timing runs
  params.config.verbosity = 0;
  params.config.compress_mode = true;
  params.config.remove_origin = false;
  params.config.recurse = true; // Assume we want to find all generated files

  // Configure OpenMP threads and nesting based on requested max
  // We will override this within specific benchmark loops if needed
  omp_set_max_active_levels(2); // Allow nesting
  // omp_set_num_threads(params.threads); // Set globally? Or per loop? Let's
  // set per loop. if (params.threads > 1) omp_set_nested(true); // Enable
  // nesting globally if p > 1

  std::cout << "--- Benchmark Initializing ---" << std::endl;
  // Update the print statement for the new type
  std::cout << "Type: " << params.type << ", Threads: " << params.threads
            << ", Iterations: " << params.iterations
            << ", Warmup: " << params.warmup;
  if (params.type == "one_large") {
    std::cout << ", File Size: " << params.large_file_size << " bytes";
  } else if (params.type == "many_small") {
    std::cout << ", Num Small Files: " << params.num_small_files << ", Sizes: ["
              << params.min_small_file_size << " - "
              << params.max_small_file_size << "] bytes";
  } else if (params.type == "many_large_sequential" ||
             params.type == "many_large_parallel" ||
             params.type == "many_large_parallel_right") { // Add new type
    static const size_t MIN_LARGE = 50ULL * 1024 * 1024;
    static const size_t MAX_LARGE = 250ULL * 1024 * 1024;
    // Use the defined constants here
    std::cout << ", Num Large Files: 10"
              << ", Size Range: [" << MIN_LARGE << " - " << MAX_LARGE
              << "] bytes";
  }
  std::cout << std::endl;
  std::cout << "Large Threshold: " << params.config.large_file_threshold
            << ", Block Size: " << params.config.block_size << std::endl;

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
      // Benchmark for many small files: sweep threads only
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      std::ofstream csv_small("results/data/benchmark_many_small.csv");
      csv_small << "threads,seq_time_s,par_time_s,speedup" << '\n';
      // Sequential baseline
      ConfigData cfg_seq = params.config;
      cfg_seq.num_threads = 1;
      omp_set_num_threads(1);
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
        cfg_p.num_threads = th;
        omp_set_num_threads(th);
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
      // Benchmark many large files: sequential file-level dispatch, nested
      // block parallelism
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      std::ofstream csv_seq("results/data/benchmark_many_large_sequential.csv");
      csv_seq << "threads,seq_time_s,par_time_s,speedup" << '\n';
      // Baseline: single-thread nested on each file
      ConfigData cfg_b = params.config;
      double base_time;
      {
        cfg_b.num_threads = 1;
        omp_set_nested(true);
        omp_set_num_threads(1);
        auto work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          for (auto &item : work_items)
            Compressor::process_file(item.path, cfg_b);
          return true;
        };
        auto r =
            BenchUtils::run_benchmark(work, params.iterations, params.warmup);
        base_time = r.median_time_s;
      }
      csv_seq << 1 << "," << base_time << "," << base_time << ",1" << '\n';
      std::cout << "ManyLargeSeq: Seq baseline=" << std::fixed
                << std::setprecision(2) << base_time << "s" << std::endl;
      std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                << std::setw(10) << "Speedup" << std::endl;
      for (int th : thread_counts) {
        if (th == 1)
          continue;
        ConfigData cfg_p = params.config;
        cfg_p.num_threads = th;
        omp_set_nested(true);
        omp_set_num_threads(th);
        auto work_p = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          for (auto &item : work_items)
            Compressor::process_file(item.path, cfg_p);
          return true;
        };
        auto rp =
            BenchUtils::run_benchmark(work_p, params.iterations, params.warmup);
        double tp = rp.median_time_s;
        double sp = base_time / tp;
        csv_seq << th << "," << base_time << "," << tp << "," << sp << '\n';
        std::cout << std::setw(8) << th << std::setw(12) << std::fixed
                  << std::setprecision(2) << tp << std::setw(10) << sp
                  << std::endl;
      }
      csv_seq.close();
      cleanup_bench_environment();
      return 0;
    }

    if (params.type == "many_large_parallel") {
      // Matrix benchmark: nested parallelism over block sizes and threads
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      // Define block sizes (1MiB to 12MiB)
      std::vector<size_t> block_sizes;
      for (int i = 1; i <= 12; ++i)
        block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);
      std::ofstream csv("results/data/benchmark_many_large_parallel.csv");
      csv << "block_size,threads,seq_time_s,par_time_s,speedup" << '\n';
      // Loop over block sizes
      for (size_t bs : block_sizes) {
        params.config.block_size = bs;
        // Sequential baseline: nested disabled, single thread
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1;
        omp_set_nested(false);
        omp_set_num_threads(1);
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          return perform_compression_work(work_items, cfg_seq);
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq << ",1"
            << '\n';
        // Print section header and table header
        std::cout << "\nBlockSize=" << (bs / (1024 * 1024))
                  << "MiB  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::endl;
        // Parallel runs per thread count >1
        for (int th : thread_counts) {
          if (th == 1)
            continue;
          ConfigData cfg_p = params.config;
          cfg_p.num_threads = th;
          omp_set_nested(true);
          omp_set_max_active_levels(2);
          omp_set_num_threads(th);
          auto par_work = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            return perform_compression_work(work_items, cfg_p);
          };
          auto pres = BenchUtils::run_benchmark(par_work, params.iterations,
                                                params.warmup);
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
      // Matrix benchmark: Controlled nested parallelism
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      std::vector<size_t> block_sizes;
      for (int i = 1; i <= 12; ++i)
        block_sizes.push_back(static_cast<size_t>(i) * 1024 * 1024);

      std::ofstream csv("results/data/benchmark_many_large_parallel_right.csv");
      csv << "block_size,threads,seq_time_s,par_time_s,speedup,t_outer,t_inner"
          << '\n';

      for (size_t bs : block_sizes) {
        params.config.block_size = bs;
        // Sequential baseline (same as others)
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1;
        omp_set_nested(false); // No nesting for baseline
        omp_set_num_threads(1);
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          // Use sequential C++ loop for baseline consistency
          for (auto &item : work_items) {
            if (!Compressor::process_file(item.path, cfg_seq))
              return false;
          }
          return true;
          // return perform_compression_work(work_items, cfg_seq); // This would
          // use OMP loop even for 1 thread
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq << ",1"
            << '\n';

        std::cout << "\nBlockSize=" << (bs / (1024 * 1024))
                  << "MiB  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s (Right Parallelism)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::setw(8) << "T_out"
                  << std::setw(8) << "T_in" << std::endl;

        // Parallel runs with controlled nesting
        for (int p : thread_counts) {
          if (p == 1)
            continue;

          // Calculate thread distribution
          int t_outer = std::min(p, num_work_items);
          // Ensure t_outer doesn't exceed p
          t_outer = std::min(t_outer, p);
          // Calculate t_inner, ensuring it's at least 1
          int t_inner = std::max(1, p / t_outer);
          // Optional: Refine t_inner to not exceed p / t_outer strictly?
          // t_inner = std::min(t_inner, p); // Ensure t_inner <= p
          // Example: p=7, nfiles=10 -> t_outer=7, t_inner=max(1, 7/7)=1.
          // Total=7*1=7 <= 7. OK. Example: p=12, nfiles=4 -> t_outer=4,
          // t_inner=max(1, 12/4)=3. Total=4*3=12 <= 12. OK. Example: p=10,
          // nfiles=4 -> t_outer=4, t_inner=max(1, 10/4)=2. Total=4*2=8 <= 10.
          // OK.

          ConfigData cfg_p_right = params.config;
          cfg_p_right.num_threads = t_inner; // Pass t_inner to inner loops

          omp_set_nested(true);
          omp_set_max_active_levels(2);
          // Set threads for the *next* outer parallel region
          omp_set_num_threads(t_outer);

          auto par_work_right = [&]() {
            TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
            // perform_compression_work will create t_outer threads.
            // Each thread calling process_file -> compress_large_file will use
            // cfg_p_right.num_threads (t_inner) for the inner parallel for.
            return perform_compression_work(work_items, cfg_p_right);
          };

          auto pres = BenchUtils::run_benchmark(
              par_work_right, params.iterations, params.warmup);
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

    // --- Fallback for one_large (if not handled above) ---
    // This block was missing the structure to iterate threads and block sizes
    // Reimplementing similar to many_large_parallel
    if (params.type == "one_large") {
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);

      // Define block sizes (1MiB to 12MiB) or use list if provided
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

        // Sequential baseline for this block size
        ConfigData cfg_seq = params.config;
        cfg_seq.num_threads = 1;
        omp_set_num_threads(1);
        omp_set_nested(false); // No nesting needed for one_large baseline
        auto seq_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          return perform_compression_work(work_items, cfg_seq);
        };
        auto res_bs = BenchUtils::run_benchmark(seq_work, params.iterations,
                                                params.warmup);
        if (!res_bs.success)
          throw std::runtime_error("one_large baseline failed");
        double time_seq = res_bs.median_time_s;
        csv << bs << "," << 1 << "," << time_seq << "," << time_seq << ",1"
            << '\n';

        std::cout << "\nBlockSize=" << (bs / (1024 * 1024))
                  << "MiB  Seq(s)=" << std::fixed << std::setprecision(2)
                  << time_seq << "s (One Large)" << std::endl;
        std::cout << std::setw(8) << "Threads" << std::setw(12) << "Par(s)"
                  << std::setw(10) << "Speedup" << std::endl;

        // Parallel runs for this block size
        for (int th : thread_counts) { // Iterate using 'th'
          if (th == 1)
            continue;

          ConfigData cfg_par =
              params.config;        // Use current block size from outer loop
          cfg_par.num_threads = th; // Set thread count for inner parallelism
          omp_set_num_threads(th); // Set threads for the parallel region inside
                                   // perform_compression_work
          omp_set_nested(false); // Nesting off for one_large block parallelism

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
