#include "bench_utils.hpp"
#include "compressor.hpp"
#include "config.hpp"
#include "file_handler.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iomanip> // For formatting output
#include <iostream>
#include <map>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// --- Benchmark Configuration ---
const std::string BENCH_DIR = "./test_data_bench_cpp";

// Structure to hold benchmark parameters
struct BenchParams { /* ... identical to test_main.cpp version ... */
  std::string type = "one_large";
  int threads = omp_get_max_threads();
  int iterations = 5;
  int warmup = 1;
  size_t large_file_size = 100 * 1024 * 1024;
  size_t small_file_size = 100 * 1024;
  int num_small_files = 100;
  ConfigData config;
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
    if (args.count("small_size"))
      params.small_file_size = std::stoull(args["small_size"]);
    if (args.count("num_small"))
      params.num_small_files = std::stoi(args["num_small"]);
    if (args.count("verbosity"))
      params.config.verbosity = std::stoi(args["verbosity"]);
    if (args.count("threshold"))
      params.config.large_file_threshold = std::stoull(args["threshold"]);
    if (args.count("blocksize"))
      params.config.block_size = std::stoull(args["blocksize"]);
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
    for (int i = 0; i < params.num_small_files; ++i) {
      const std::string file_path =
          std::string(BENCH_DIR) + "/small_file_" + std::to_string(i) + ".bin";
      if (!TestUtils::create_random_file(file_path, params.small_file_size,
                                         0)) {
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
            << ", Warmup: " << params.warmup << std::endl;
  std::cout << "Large Threshold: " << params.config.large_file_threshold
            << ", Block Size: " << params.config.block_size << std::endl;

  try {
    setup_bench_environment(params);
    std::vector<std::string> initial_paths = {BENCH_DIR};
    auto work_items =
        FileHandler::discover_work_items(initial_paths, params.config);
    if (work_items.empty())
      throw std::runtime_error("No work items found.");

    // --- Sequential Run ---
    std::cout << "\nRunning Sequential..." << std::flush;
    ConfigData cfg_seq = params.config; // Copy config
    cfg_seq.num_threads = 1;
    omp_set_num_threads(1);
    auto seq_work = [&]() -> bool {
      TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
      return perform_compression_work(work_items, cfg_seq);
    };
    double time_seq_s =
        BenchUtils::run_benchmark(seq_work, params.iterations, params.warmup);
    if (time_seq_s < 0)
      throw std::runtime_error("Sequential benchmark failed.");
    std::cout << " Done. Time: " << std::fixed << std::setprecision(4)
              << time_seq_s << " s" << std::endl;

    // --- Parallel Run ---
    std::cout << "Running Parallel (" << params.threads << " threads)..."
              << std::flush;
    ConfigData cfg_par = params.config; // Copy config
    cfg_par.num_threads = params.threads;
    omp_set_num_threads(params.threads);
    auto par_work = [&]() -> bool {
      TestUtils::clean_files_with_suffix(BENCH_DIR, SUFFIX, true, 0);
      return perform_compression_work(work_items, cfg_par);
    };
    double time_par_s =
        BenchUtils::run_benchmark(par_work, params.iterations, params.warmup);
    if (time_par_s < 0)
      throw std::runtime_error("Parallel benchmark failed.");
    std::cout << " Done. Time: " << std::fixed << std::setprecision(4)
              << time_par_s << " s" << std::endl;

    // --- Results ---
    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "Type:       " << params.type << std::endl;
    std::cout << "Threads:    " << params.threads << std::endl;
    std::cout << "Iterations: " << params.iterations << " (+" << params.warmup
              << " warmup)" << std::endl;
    std::cout << "Seq Time:   " << std::fixed << std::setprecision(4)
              << time_seq_s << " s" << std::endl;
    std::cout << "Par Time:   " << std::fixed << std::setprecision(4)
              << time_par_s << " s" << std::endl;
    if (time_par_s > 1e-9) {
      double speedup = time_seq_s / time_par_s;
      std::cout << "Speedup:    " << std::fixed << std::setprecision(2)
                << speedup << "x" << std::endl;
    } else {
      std::cout << "Speedup:    N/A" << std::endl;
    }
    std::cout << "------------------------" << std::endl;

    cleanup_bench_environment();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n!!! Benchmark FAILED: " << e.what() << std::endl;
    cleanup_bench_environment();
    return 1;
  }
}
