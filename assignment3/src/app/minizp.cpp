#include "cmdline.hpp"
#include "compressor.hpp"
#include "config.hpp"
#include "file_handler.hpp"

#include <atomic>
#include <iostream>
#include <omp.h> // Include for OpenMP functions
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  ConfigData config;
  std::vector<std::string> initial_paths;

  // 1. Parse Command Line
  if (!CmdLine::parseCommandLine(argc, argv, config, initial_paths)) {
    return 1; // Exit if parsing failed or help was shown
  }

  if (config.verbosity >= 2) {
    std::cout << "Configuration:\n"
              << "  Mode:          "
              << (config.compress_mode ? "Compress" : "Decompress") << "\n"
              << "  Remove Origin: " << (config.remove_origin ? "Yes" : "No")
              << "\n"
              << "  Recurse:       " << (config.recurse ? "Yes" : "No") << "\n"
              << "  Verbosity:     " << config.verbosity << "\n"
              << "  Threads:       " << config.num_threads << "\n"
              << "  Large Thresh:  " << config.large_file_threshold << "\n"
              << "  Block Size:    " << config.block_size << std::endl;
  }

  // 2. Discover Files/Work Items
  if (config.verbosity >= 2) {
    std::cout << "Discovering work items..." << std::endl;
  }
  std::vector<FileHandler::WorkItem> work_items;
  try {
    work_items = FileHandler::discover_work_items(initial_paths, config);
  } catch (const std::exception &e) {
    std::cerr << "Error during file discovery: " << e.what() << std::endl;
    return 1;
  }

  if (config.verbosity >= 2) {
    std::cout << "Found " << work_items.size() << " items to process."
              << std::endl;
  }
  if (work_items.empty()) {
    std::cout << "No files found matching the criteria." << std::endl;
    return 0;
  }

  // 3. Process Files in Parallel
  std::atomic<bool> processing_error = false;
  double start_time = omp_get_wtime();

#pragma omp parallel for default(none)                                         \
    shared(work_items, config, processing_error, std::cerr, std::cout)         \
    schedule(dynamic)
  for (size_t i = 0; i < work_items.size(); ++i) {
    // Check if an error occurred in another thread to potentially stop early
    // (optional)
    if (processing_error.load()) {
      continue; // Skip remaining work if a critical error happened
    }

    const auto &item = work_items[i];
    // Compressor instance per thread if it has state, or shared if
    // stateless/thread-safe Assuming stateless for now:
    bool success = false;

    try {
      if (config.compress_mode) {
        if (config.verbosity >= 2) {
#pragma omp critical(cout_lock)
          {
            std::cout << "[Thread " << omp_get_thread_num()
                      << "] Compressing: " << item.path
                      << " (Size: " << item.size << ")" << std::endl;
          }
        }
        success = Compressor::process_file(item.path, config);
      } else {
        if (config.verbosity >= 2) {
#pragma omp critical(cout_lock)
          {
            std::cout << "[Thread " << omp_get_thread_num()
                      << "] Decompressing: " << item.path << std::endl;
          }
        }
        success = Compressor::decompress_file(item.path, config);
      }

      if (!success) {
        processing_error = true; // Signal error
#pragma omp critical(cerr_lock)
        {
          std::cerr << "[Thread " << omp_get_thread_num()
                    << "] Error processing: " << item.path << std::endl;
        }
      } else if (config.verbosity >= 2) {
#pragma omp critical(cout_lock)
        {
          std::cout << "[Thread " << omp_get_thread_num()
                    << "] Done: " << item.path << std::endl;
        }
      }

    } catch (const std::exception &e) {
      processing_error = true; // Signal error
#pragma omp critical(cerr_lock)
      {
        std::cerr << "[Thread " << omp_get_thread_num()
                  << "] Exception processing " << item.path << ": " << e.what()
                  << std::endl;
      }
    } catch (...) {
      processing_error = true; // Signal error
#pragma omp critical(cerr_lock)
      {
        std::cerr << "[Thread " << omp_get_thread_num()
                  << "] Unknown exception processing " << item.path
                  << std::endl;
      }
    }

  } // --- End parallel for ---

  double end_time = omp_get_wtime();

  // 4. Final Report
  if (config.verbosity >= 1) {
    std::cout << "--------------------\n";
    std::cout << "Processing finished in " << end_time - start_time
              << " seconds." << std::endl;
    if (processing_error) {
      std::cout << "Exiting with Errors." << std::endl;
    } else {
      std::cout << "Exiting with Success." << std::endl;
    }
  }

  return processing_error ? 1 : 0;
}
