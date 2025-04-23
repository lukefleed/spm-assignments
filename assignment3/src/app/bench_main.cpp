#include "bench_utils.hpp"
#include "compressor.hpp"
#include "config.hpp"
#include "file_handler.hpp"
#include "test_utils.hpp"

#include <atomic>
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
const std::string BENCH_DIR = "./test_data_bench_cpp";

// Structure to hold benchmark parameters
struct BenchParams {
  std::string type = "one_large";
  int threads = omp_get_max_threads();
  int iterations = 2;
  int warmup = 1;
  size_t large_file_size = 512 * 1024 * 1024;
  // size_t small_file_size = 100 * 1024; // Removed: sizes now random and
  // distinct
  int num_small_files = 4000;
  ConfigData config;
  std::vector<size_t> block_sizes_list; // List of block sizes for sweeping
};

// --- Helper: Parse Args ---
bool parseBenchArgs(
    int argc, char *argv[],
    BenchParams &params) { /* ... identical to test_main.cpp version ... */
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
    // Validations
    if (params.type != "one_large" && params.type != "many_small")
      throw std::runtime_error("Invalid type");
    if (params.threads <= 0)
      throw std::runtime_error("Threads must be positive");
    // ... other validations
  } catch (const std::exception &e) {
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return false;
  }
  return true;
}

// --- Helper: Setup ---
void setup_bench_environment(
    const BenchParams
        &params) { /* ... identical to test_main.cpp version ... */
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
  } else { // many_small
    // Generate many small files with random, distinct sizes between 1KB and
    // 50KB
    std::random_device rd;
    std::mt19937_64 gen(rd());
    constexpr size_t min_size = 1 * 1024;  // 1 KB
    constexpr size_t max_size = 50 * 1024; // 50 KB
    std::uniform_int_distribution<size_t> dist(min_size, max_size);
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

// --- Helper: Cleanup ---
void cleanup_bench_environment() { /* ... identical to test_main.cpp version ...
                                    */
  std::cout << "Cleaning up benchmark environment..." << std::endl;
  std::error_code ec;
  fs::remove_all(BENCH_DIR, ec);
}

// --- Compression Work Function ---
bool perform_compression_work(
    const std::vector<FileHandler::WorkItem> &items,
    ConfigData &cfg) { /* ... identical to test_main.cpp version ... */
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

// --- Main ---
int main(int argc, char *argv[]) {
  BenchParams params;
  if (!parseBenchArgs(argc, argv, params))
    return 1;

  // Ensure low verbosity for timing runs
  params.config.verbosity = 0;
  params.config.compress_mode = true;
  params.config.remove_origin = false;
  params.config.recurse = true; // Assume we want to find all generated files

  std::cout << "--- Benchmark Initializing ---" << std::endl;
  std::cout << "Type: " << params.type << ", Threads: " << params.threads
            << ", Iterations: " << params.iterations
            << ", Warmup: " << params.warmup;
  if (params.type == "one_large") {
    std::cout << ", File Size: " << params.large_file_size << " bytes";
  } else {
    std::cout << ", Num Small Files: " << params.num_small_files
              << ", Sizes: random < " << params.config.large_file_threshold
              << " bytes";
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

    // --- Automated Benchmark Sweep ---
    if (params.type == "many_small") {
      // Benchmark for many small files: sweep threads only
      std::vector<int> thread_counts;
      for (int t = 1; t <= params.threads; ++t)
        thread_counts.push_back(t);
      std::ofstream csv_small("benchmark_many_small.csv");
      csv_small << "threads,seq_time_s,par_time_s,speedup" << '\n';
      // Sequential baseline
      ConfigData cfg_seq = params.config;
      cfg_seq.num_threads = 1;
      omp_set_num_threads(1);
      auto seq_work_small = [&]() {
        TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
        return perform_compression_work(work_items, cfg_seq);
      };
      double time_seq = BenchUtils::run_benchmark(
          seq_work_small, params.iterations, params.warmup);
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
        double time_par = BenchUtils::run_benchmark(
            par_work_small, params.iterations, params.warmup);
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

    // Hard-coded block sizes for sweep (1MiB to 12MiB)
    std::vector<size_t> block_sizes = {
        1 * 1024 * 1024, 2 * 1024 * 1024,  3 * 1024 * 1024,  4 * 1024 * 1024,
        5 * 1024 * 1024, 6 * 1024 * 1024,  7 * 1024 * 1024,  8 * 1024 * 1024,
        9 * 1024 * 1024, 10 * 1024 * 1024, 11 * 1024 * 1024, 12 * 1024 * 1024};
    // Determine thread counts to test (1..max)
    std::vector<int> thread_counts;
    for (int t = 1; t <= params.threads; ++t)
      thread_counts.push_back(t);
    // Open CSV file for results
    std::ofstream csv("benchmark_matrix_results.csv");
    csv << "block_size,threads,seq_time_s,par_time_s,speedup" << '\n';

    for (size_t bs : block_sizes) {
      // Set block size in configuration
      params.config.block_size = bs;

      // --- Sequential baseline (once per block size) ---
      ConfigData cfg_seq = params.config;
      cfg_seq.num_threads = 1;
      omp_set_num_threads(1);
      omp_set_nested(0);
      auto seq_work = [&]() {
        TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
        return perform_compression_work(work_items, cfg_seq);
      };
      double time_seq_s =
          BenchUtils::run_benchmark(seq_work, params.iterations, params.warmup);
      // Section header for this block size with sequential baseline
      std::cout << std::endl
                << "BlockSize=" << bs << "  Seq(s)=" << std::fixed
                << std::setprecision(2) << time_seq_s << "s" << std::endl;
      // Table header for parallel runs
      std::cout << std::setw(10) << "Threads" << std::setw(12) << "Par(s)"
                << std::setw(10) << "Speedup" << std::endl;

      // Record baseline CSV
      csv << bs << "," << 1 << "," << time_seq_s << "," << time_seq_s << ",1"
          << '\n';

      // Sweep parallel runs for threads > 1
      for (int th : thread_counts) {
        if (th == 1)
          continue; // skip baseline case
        ConfigData cfg_par = params.config;
        cfg_par.num_threads = th;
        omp_set_num_threads(th);
        omp_set_nested(th > 1);
        auto par_work = [&]() {
          TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
          return perform_compression_work(work_items, cfg_par);
        };
        double time_par_s = BenchUtils::run_benchmark(
            par_work, params.iterations, params.warmup);
        double speedup = time_seq_s / time_par_s;
        // Write CSV and print parallel row
        csv << bs << "," << th << "," << time_seq_s << "," << time_par_s << ","
            << speedup << '\n';
        std::cout << std::setw(10) << th << std::setw(12) << std::fixed
                  << std::setprecision(2) << time_par_s << std::setw(10)
                  << std::fixed << std::setprecision(2) << speedup << std::endl;
      }
    }
    csv.close();
    cleanup_bench_environment();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n!!! Benchmark FAILED: " << e.what() << std::endl;
    cleanup_bench_environment();
    return 1;
  }
}
