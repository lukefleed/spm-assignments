/**
 * @file test_main.cpp
 * @brief Unit tests for verifying compressor correctness.
 *
 * Sets up test environments, runs compression and decompression tests,
 * edge cases, and reports pass/fail status for sequential and parallel modes.
 */

#include "compressor.hpp"
#include "config.hpp"
#include "file_handler.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <cstdlib> // exit
#include <filesystem>
#include <iostream>
#include <map>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int BANNER_WIDTH = 60;

// --- Test Configuration ---
const std::string TEST_DIR = "./test_data_correctness_cpp";
const std::string ORIGINALS_DIR =
    "./test_data_originals_cpp"; // Sibling directory
const std::map<std::string, size_t> FILES_TO_CREATE = {
    {"small1.bin", 1 * 1024},
    {"small2.bin", 100 * 1024},
    {"medium.bin", 5 * 1024 * 1024},
    {"large.bin", 30 * 1024 * 1024},
    {"zero.bin", 0},
    {"subdir/small3.bin", 10 * 1024}};
const int TEST_VERBOSITY = 1;

// Forward declarations for test functions
void test_compression_phase(const ConfigData &cfg);
void test_decompression_phase(const ConfigData &cfg);
void test_edge_case_remove_origin_compress(const ConfigData &cfg);
void test_edge_case_remove_origin_decompress(const ConfigData &cfg);
void test_edge_case_invalid_input(const ConfigData &cfg);
void test_edge_case_threshold_override(const ConfigData &cfg);
void test_edge_case_recursion(const ConfigData &cfg);

/// @brief Set up or recreate test directory and sample files
/// @param create_test_files If true, copy originals into test dir for testing
void setup_test_environment(bool create_test_files) {
  std::cout << "Setting up test environment (Test: " << TEST_DIR
            << ", Originals: " << ORIGINALS_DIR << ")..." << std::endl;
  std::error_code ec;

  // Remove previous directories
  fs::remove_all(TEST_DIR, ec);
  fs::remove_all(ORIGINALS_DIR, ec);

  // Create originals directory structure
  fs::create_directories(ORIGINALS_DIR + "/subdir", ec);
  if (ec)
    throw std::runtime_error("Failed originals dir creation: " + ec.message());

  // Create original random files
  for (const auto &pair : FILES_TO_CREATE) {
    const std::string relative_path = pair.first;
    const size_t size = pair.second;
    const std::string full_path_orig = ORIGINALS_DIR + "/" + relative_path;
    // Ensure parent directory exists within ORIGINALS_DIR
    fs::create_directories(fs::path(full_path_orig).parent_path(), ec);
    if (ec)
      throw std::runtime_error("Failed creating parent dir in originals: " +
                               ec.message());

    if (!TestUtils::create_random_file(full_path_orig, size, TEST_VERBOSITY)) {
      throw std::runtime_error("Failed creation: " + full_path_orig);
    }
  }

  // If requested, create test directory and copy originals into it
  if (create_test_files) {
    fs::create_directories(TEST_DIR + "/subdir", ec);
    if (ec)
      throw std::runtime_error("Failed test dir creation: " + ec.message());

    for (const auto &pair : FILES_TO_CREATE) {
      const std::string relative_path = pair.first;
      const std::string full_path_orig = ORIGINALS_DIR + "/" + relative_path;
      const std::string full_path_test = TEST_DIR + "/" + relative_path;
      // Ensure parent directory exists within TEST_DIR
      fs::create_directories(fs::path(full_path_test).parent_path(), ec);
      if (ec)
        throw std::runtime_error("Failed creating parent dir in test: " +
                                 ec.message());

      fs::copy_file(full_path_orig, full_path_test,
                    fs::copy_options::overwrite_existing, ec);
      if (ec)
        throw std::runtime_error("Failed copy from originals to test: " +
                                 ec.message());
    }
  }
  std::cout << "Setup complete." << std::endl;
}

/// @brief Remove the test environment directory and its contents
void cleanup_test_environment() {
  std::cout << "Cleaning up test environment..." << std::endl;
  std::error_code ec;
  fs::remove_all(TEST_DIR, ec);
  fs::remove_all(ORIGINALS_DIR, ec); // Also remove originals dir
}

/// @brief Tests the compression phase
/// @param cfg Configuration parameters
void test_compression_phase(const ConfigData &cfg_base) {
  std::cout << "\n[Compression Phase]" << std::endl;
  setup_test_environment(true); // Ensure originals exist in test dir
  TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                     TEST_VERBOSITY); // Clean old zips

  // Generate original checksum file
  if (cfg_base.verbosity >= 1)
    std::cout << "Generating original checksum file..." << std::endl;
  {
    std::string orig_md5 = TEST_DIR + "/orig.md5";
    std::string cmd = "md5sum";
    for (const auto &pair : FILES_TO_CREATE) {
      cmd += " \"" + TEST_DIR + "/" + pair.first + "\"";
    }
    cmd += " > \"" + orig_md5 + "\"";
    if (std::system(cmd.c_str()) != 0)
      throw std::runtime_error("Failed to generate original checksum file");
  }

  ConfigData cfg = cfg_base; // Local copy to modify
  cfg.compress_mode = true;
  cfg.remove_origin = false;
  std::vector<std::string> paths_to_compress = {TEST_DIR};
  auto items_to_compress =
      FileHandler::discover_work_items(paths_to_compress, cfg);

  std::atomic<bool> compress_error = false;
#pragma omp parallel for if (cfg.num_threads > 1) default(none)                \
    shared(items_to_compress, cfg, compress_error, std::cerr)                  \
    schedule(dynamic)
  for (size_t i = 0; i < items_to_compress.size(); ++i) {
    if (compress_error.load())
      continue;
    if (!Compressor::process_file(items_to_compress[i].path, cfg)) {
      compress_error = true; // Signal failure
    }
  }
  if (compress_error)
    throw std::runtime_error("Compression phase failed.");

  // Verify zip files exist
  for (const auto &item : items_to_compress) {
    if (!fs::exists(item.path + SUFFIX))
      throw std::runtime_error("Missing compressed file: " + item.path +
                               SUFFIX);
  }
  std::cout << "Compression verification successful." << std::endl;
}

/// @brief Tests the decompression phase and content verification
/// @param cfg Configuration parameters
void test_decompression_phase(const ConfigData &cfg_base) {
  std::cout << "\n[Decompression Phase]" << std::endl;
  // Assume compression phase ran, zips exist, originals are in ORIGINALS_DIR
  // Remove original .bin files from TEST_DIR before decompressing
  TestUtils::clean_files_with_suffix(TEST_DIR, ".bin", true,
                                     TEST_VERBOSITY); // Adjust suffix if needed

  ConfigData cfg = cfg_base; // Local copy to modify
  cfg.compress_mode = false;
  cfg.remove_origin = false;
  std::vector<std::string> paths_to_decompress = {TEST_DIR};
  auto items_to_decompress =
      FileHandler::discover_work_items(paths_to_decompress, cfg);

  std::atomic<bool> decompress_error = false;
#pragma omp parallel for if (cfg.num_threads > 1) default(none)                \
    shared(items_to_decompress, cfg, decompress_error, std::cerr)              \
    schedule(dynamic)
  for (size_t i = 0; i < items_to_decompress.size(); ++i) {
    if (decompress_error.load())
      continue;
    if (!Compressor::decompress_file(items_to_decompress[i].path, cfg)) {
      decompress_error = true; // Signal failure
    }
  }
  if (decompress_error)
    throw std::runtime_error("Decompression phase failed.");

  // Verify decompressed files exist and match originals
  std::cout << "Verifying decompressed content..." << std::endl;
  for (const auto &pair : FILES_TO_CREATE) {
    const std::string relative_path = pair.first;
    const std::string original_path = ORIGINALS_DIR + "/" + relative_path;
    const std::string decompressed_path = TEST_DIR + "/" + relative_path;
    if (!fs::exists(decompressed_path))
      throw std::runtime_error("Missing decompressed file: " +
                               decompressed_path);
    if (!TestUtils::compare_files(original_path, decompressed_path,
                                  TEST_VERBOSITY)) {
      throw std::runtime_error("Content mismatch: " + decompressed_path +
                               " vs " + original_path);
    }
  }
  std::cout << "Decompression verification successful." << std::endl;

  // Generate regenerated checksum file and compare with original
  if (cfg.verbosity >= 1)
    std::cout << "Generating regenerated checksum file and comparing..."
              << std::endl;
  {
    std::string regen_md5 = TEST_DIR + "/regen.md5";
    std::string cmd = "md5sum";
    for (const auto &pair : FILES_TO_CREATE) {
      cmd += " \"" + TEST_DIR + "/" + pair.first + "\"";
    }
    cmd += " > \"" + regen_md5 + "\"";
    if (std::system(cmd.c_str()) != 0)
      throw std::runtime_error("Failed to generate regenerated checksum file");
    std::string diff_cmd =
        "diff -q \"" + TEST_DIR + "/orig.md5\" \"" + regen_md5 + "\"";
    if (std::system(diff_cmd.c_str()) != 0)
      throw std::runtime_error("Checksum files differ (orig vs regen)");
  }
}

/// @brief Tests the remove_origin flag during compression
/// @param cfg Configuration parameters
void test_edge_case_remove_origin_compress(const ConfigData &cfg_base) {
  std::cout << "\n[Edge Case: remove_origin compress]" << std::endl;
  if (cfg_base.verbosity >= 1)
    std::cout << "Testing compression with remove_origin flag..." << std::endl;
  // Reset environment and clean zips
  setup_test_environment(true);
  TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true, TEST_VERBOSITY);
  // Configure remove_origin
  ConfigData cfg = cfg_base;
  cfg.compress_mode = true;
  cfg.remove_origin = true;
  auto items = FileHandler::discover_work_items({TEST_DIR}, cfg);
  // Compress (using sequential loop for simplicity in test case)
  for (const auto &it : items) {
    if (!Compressor::process_file(it.path, cfg))
      throw std::runtime_error(
          "remove_origin compress test: compression failed for " + it.path);
  }
  // Verify originals deleted, zips exist
  for (const auto &pair : FILES_TO_CREATE) {
    const std::string orig = TEST_DIR + "/" + pair.first;
    const std::string zipf = orig + SUFFIX;
    if (std::filesystem::exists(orig))
      throw std::runtime_error(
          "remove_origin compress test: original still exists: " + orig);
    if (!std::filesystem::exists(zipf))
      throw std::runtime_error("remove_origin compress test: zip missing: " +
                               zipf);
  }
  std::cout << "remove_origin compress edge case passed." << std::endl;
}

/// @brief Tests the remove_origin flag during decompression
/// @param cfg Configuration parameters
void test_edge_case_remove_origin_decompress(const ConfigData &cfg_base) {
  std::cout << "\n[Edge Case: remove_origin decompress]" << std::endl;
  // Setup and compress normally first
  setup_test_environment(true);
  TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true, TEST_VERBOSITY);
  ConfigData cfg_compress = cfg_base;
  cfg_compress.compress_mode = true;
  cfg_compress.remove_origin = false;
  auto items_compress =
      FileHandler::discover_work_items({TEST_DIR}, cfg_compress);
  for (const auto &it : items_compress) {
    if (!Compressor::process_file(it.path, cfg_compress))
      throw std::runtime_error(
          "remove_origin decompress setup: compression failed for " + it.path);
  }
  // Decompress with remove_origin=true
  ConfigData cfg_decompress = cfg_base;
  cfg_decompress.compress_mode = false;
  cfg_decompress.remove_origin = true;
  auto items_decompress =
      FileHandler::discover_work_items({TEST_DIR}, cfg_decompress);
  for (const auto &it : items_decompress) {
    if (!Compressor::decompress_file(it.path, cfg_decompress))
      throw std::runtime_error(
          "remove_origin decompress test: decompression failed for " + it.path);
  }
  // Verify .zip removed and originals exist
  for (const auto &pair : FILES_TO_CREATE) {
    std::string zipf = TEST_DIR + "/" + pair.first + SUFFIX;
    std::string origf = TEST_DIR + "/" + pair.first;
    if (fs::exists(zipf))
      throw std::runtime_error(
          "remove_origin decompress test: zip still exists: " + zipf);
    if (!fs::exists(origf))
      throw std::runtime_error(
          "remove_origin decompress test: original missing: " + origf);
  }
  std::cout << "remove_origin decompress edge case passed." << std::endl;
}

/// @brief Tests handling of invalid input paths
/// @param cfg Configuration parameters
void test_edge_case_invalid_input(const ConfigData &cfg) {
  std::cout << "\n[Edge Case: invalid input path]" << std::endl;
  std::string badpath = TEST_DIR + "/nonexistent.bin";
  bool failure_correctly_handled = false;

  try {
    // Ensure the test directory exists, but don't create sample files
    setup_test_environment(false);
    ConfigData cfg_compress = cfg;
    cfg_compress.compress_mode = true;

    // Attempt to process the non-existent file
    if (!Compressor::process_file(badpath, cfg_compress)) {
      // process_file returned false, which is an expected behavior for a
      // non-existent file.
      std::cout << "Invalid input path correctly handled (process_file "
                   "returned false)."
                << std::endl;
      failure_correctly_handled = true;
    } else {
      // process_file returned true, which is incorrect for a non-existent file.
      std::cerr
          << "Error: process_file unexpectedly succeeded for invalid path: "
          << badpath << std::endl;
      // Let the final check below throw the test failure error.
      failure_correctly_handled = false;
    }
  } catch (const std::exception &e) {
    // process_file threw an exception. This is also an acceptable way to handle
    // the error.
    std::cout << "Invalid input path correctly handled (process_file threw "
                 "exception: "
              << e.what() << ")." << std::endl;
    failure_correctly_handled = true;
  } catch (...) {
    // process_file threw an unknown exception. Also acceptable error handling.
    std::cout << "Invalid input path correctly handled (process_file threw "
                 "unknown exception)."
              << std::endl;
    failure_correctly_handled = true;
  }

  // Final check: If the failure wasn't handled correctly (either by returning
  // false or throwing), fail the test.
  if (!failure_correctly_handled) {
    throw std::runtime_error(
        "invalid input path test: Failure for non-existent file was not "
        "handled correctly (process_file neither returned false nor threw an "
        "exception).");
  }

  std::cout << "invalid input path edge case passed." << std::endl;
}

/// @brief Tests overriding the large file threshold
/// @param cfg Configuration parameters
void test_edge_case_threshold_override(const ConfigData &cfg_base) {
  std::cout << "\n[Edge Case: threshold override]" << std::endl;
  setup_test_environment(true);
  TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true, TEST_VERBOSITY);
  ConfigData cfg = cfg_base;
  cfg.compress_mode = true;
  cfg.remove_origin = false;
  cfg.large_file_threshold = 1; // Force all files to large path
  auto items_compress = FileHandler::discover_work_items({TEST_DIR}, cfg);
  for (const auto &it : items_compress) {
    if (!Compressor::process_file(it.path, cfg))
      throw std::runtime_error(
          "threshold override test: compression failed for " + it.path);
  }
  // Decompress normally
  cfg.compress_mode = false;
  auto items_decompress = FileHandler::discover_work_items({TEST_DIR}, cfg);
  for (const auto &it : items_decompress) {
    if (!Compressor::decompress_file(it.path, cfg))
      throw std::runtime_error(
          "threshold override test: decompression failed for " + it.path);
  }
  // Compare content
  for (const auto &pair : FILES_TO_CREATE) {
    std::string orig = ORIGINALS_DIR + "/" + pair.first;
    std::string decomp = TEST_DIR + "/" + pair.first;
    if (!TestUtils::compare_files(orig, decomp, TEST_VERBOSITY))
      throw std::runtime_error("threshold override content mismatch: " +
                               decomp);
  }
  std::cout << "threshold override edge case passed." << std::endl;
}

/// @brief Tests recursive vs non-recursive discovery
/// @param cfg Configuration parameters
void test_edge_case_recursion(const ConfigData &cfg_base) {
  std::cout << "\n[Edge Case: recursion disabled/enabled]" << std::endl;
  setup_test_environment(true);
  TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true, TEST_VERBOSITY);

  // Config without recursion
  ConfigData cfg_no_recurse = cfg_base;
  cfg_no_recurse.compress_mode = true;
  cfg_no_recurse.remove_origin = false;
  cfg_no_recurse.recurse = false;
  auto items_nr = FileHandler::discover_work_items({TEST_DIR}, cfg_no_recurse);
  size_t expected_nr = 0;
  for (const auto &pair : FILES_TO_CREATE) {
    if (pair.first.find('/') == std::string::npos) {
      expected_nr++;
    }
  }
  if (items_nr.size() != expected_nr)
    throw std::runtime_error("recursion disabled test: expected " +
                             std::to_string(expected_nr) + " items, got " +
                             std::to_string(items_nr.size()));
  for (auto &it : items_nr) {
    if (it.path.find("subdir/") != std::string::npos)
      throw std::runtime_error("recursion disabled test: found subdir item " +
                               it.path);
  }
  std::cout << "Recursion disabled check passed." << std::endl;

  // Config with recursion
  ConfigData cfg_recurse = cfg_base;
  cfg_recurse.compress_mode = true;
  cfg_recurse.recurse = true;
  auto items_r = FileHandler::discover_work_items({TEST_DIR}, cfg_recurse);
  size_t expected_r = FILES_TO_CREATE.size();
  if (items_r.size() != expected_r)
    throw std::runtime_error("recursion enabled test: expected " +
                             std::to_string(expected_r) + " items, got " +
                             std::to_string(items_r.size()));
  std::cout << "Recursion enabled check passed." << std::endl;
  std::cout << "recursion edge case passed." << std::endl;
}

/// @brief Execute correctness tests under given configuration
/// @param cfg Configuration parameters for compression/decompression
/// @return true if all tests pass, false otherwise
bool run_tests(ConfigData cfg) { // Pass config by value to modify locally
  bool all_passed = true;

  // Stylized banner for test mode
  std::cout << "\n" << std::string(BANNER_WIDTH, '=') << std::endl;
  std::cout << "  Test Mode : "
            << (cfg.num_threads == 1 ? "Sequential" : "Parallel") << std::endl;
  std::cout << "  Threads   : " << cfg.num_threads << std::endl;
  std::cout << std::string(BANNER_WIDTH, '=') << std::endl;

  try {
    // --- Compression Phase ---
    test_compression_phase(cfg);

    // --- Decompression Phase ---
    test_decompression_phase(cfg);

    // --- Edge Case: remove_origin compress ---
    test_edge_case_remove_origin_compress(cfg);

    // --- Edge Case: remove_origin decompress ---
    test_edge_case_remove_origin_decompress(cfg);

    // --- Edge Case: invalid input path ---
    test_edge_case_invalid_input(cfg);

    // --- Edge Case: threshold override ---
    test_edge_case_threshold_override(cfg);

    // --- Edge Case: recursion disabled/enabled ---
    test_edge_case_recursion(cfg);

  } catch (const std::exception &e) {
    std::cerr << "!!! Test FAILED: " << e.what() << std::endl;
    all_passed = false;
  }

  cleanup_test_environment(); // Cleanup after tests for this mode

  // Finish banner
  std::cout << std::string(BANNER_WIDTH, '=') << std::endl;
  std::cout << "  Tests Completed ("
            << (cfg.num_threads == 1 ? "Sequential" : "Parallel") << ")"
            << std::endl;
  std::cout << std::string(BANNER_WIDTH, '=') << std::endl;

  return all_passed;
}

/// @brief Entry point for the test driver application.
///
/// Parses mode argument, configures threading, runs all tests,
/// and returns exit code indicating success or failure.
/// @param argc Number of command-line arguments.
/// @param argv Array of command-line argument strings.
/// @return 0 if all tests pass, non-zero otherwise.
int main(int argc, char *argv[]) {
  std::string mode = "seq"; // Default to sequential if no arg
  if (argc > 1) {
    std::string arg1 = argv[1];
    if (arg1 == "--mode=seq")
      mode = "seq";
    else if (arg1 == "--mode=par")
      mode = "par";
    else {
      std::cerr << "Usage: " << argv[0] << " [--mode=seq | --mode=par]"
                << std::endl;
      return 1;
    }
  }

  ConfigData config;
  config.verbosity = TEST_VERBOSITY;
  config.recurse = true; // Ensure recursion is tested

  if (mode == "seq") {
    config.num_threads = 1;
    omp_set_num_threads(1);
  } else {                                      // mode == "par"
    config.num_threads = omp_get_max_threads(); // Use default max threads
    if (config.num_threads <= 1) {
      std::cerr << "Warning: Parallel test mode requested, but only <= 1 "
                   "thread available. Running sequentially."
                << std::endl;
      config.num_threads = 1; // Fallback to seq if only 1 core detected
    }
    omp_set_num_threads(config.num_threads);
  }

  if (run_tests(config)) {
    std::cout << "\n*** Mode [" << mode << "] PASSED ***" << std::endl;
    return 0;
  } else {
    std::cout << "\n*** Mode [" << mode << "] FAILED ***" << std::endl;
    return 1;
  }
}
