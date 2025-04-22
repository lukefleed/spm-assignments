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

// --- Test Configuration ---
const std::string TEST_DIR = "./test_data_correctness_cpp";
const std::string ORIGINALS_DIR = TEST_DIR + "/originals";
const std::map<std::string, size_t> FILES_TO_CREATE = {
    {"small1.bin", 1 * 1024},
    {"small2.bin", 100 * 1024},
    {"medium.bin", 5 * 1024 * 1024},
    {"large.bin", 30 * 1024 * 1024},
    {"zero.bin", 0},
    {"subdir/small3.bin", 10 * 1024}};
const int TEST_VERBOSITY = 1;

// --- Helper: Setup ---
void setup_test_environment(bool create_test_files) {
  std::cout << "Setting up test environment in " << TEST_DIR << "..."
            << std::endl;
  std::error_code ec;
  fs::remove_all(TEST_DIR, ec);
  fs::create_directories(ORIGINALS_DIR + "/subdir", ec);
  if (ec)
    throw std::runtime_error("Failed test dir creation: " + ec.message());

  for (const auto &pair : FILES_TO_CREATE) {
    const std::string relative_path = pair.first;
    const size_t size = pair.second;
    const std::string full_path_orig = ORIGINALS_DIR + "/" + relative_path;
    fs::create_directories(fs::path(full_path_orig).parent_path(),
                           ec); // Ensure parent exists
    if (!TestUtils::create_random_file(full_path_orig, size, TEST_VERBOSITY)) {
      throw std::runtime_error("Failed creation: " + full_path_orig);
    }
    if (create_test_files) { // Only copy/create in main test dir if needed
                             // initially
      const std::string full_path_test = TEST_DIR + "/" + relative_path;
      fs::create_directories(fs::path(full_path_test).parent_path(), ec);
      fs::copy_file(full_path_orig, full_path_test,
                    fs::copy_options::overwrite_existing, ec);
      if (ec)
        throw std::runtime_error("Failed copy: " + ec.message());
    }
  }
  std::cout << "Setup complete." << std::endl;
}

// --- Helper: Cleanup ---
void cleanup_test_environment() {
  std::cout << "Cleaning up test environment..." << std::endl;
  std::error_code ec;
  fs::remove_all(TEST_DIR, ec);
}

// --- Test Logic ---
bool run_tests(ConfigData cfg) { // Pass config by value to modify locally
  bool all_passed = true;

  // Stylized banner for test mode
  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << "  Test Mode : "
            << (cfg.num_threads == 1 ? "Sequential" : "Parallel") << std::endl;
  std::cout << "  Threads   : " << cfg.num_threads << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  try {
    // --- Compression Phase ---
    std::cout << "\n[Compression Phase]" << std::endl;
    setup_test_environment(true); // Ensure originals exist in test dir
    TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                       TEST_VERBOSITY); // Clean old zips

    // Generate original checksum file
    if (cfg.verbosity >= 1)
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

    // --- Decompression Phase ---
    std::cout << "\n[Decompression Phase]" << std::endl;
    // Remove original .bin files before decompressing
    TestUtils::clean_files_with_suffix(
        TEST_DIR, ".bin", true, TEST_VERBOSITY); // Adjust suffix if needed

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
        throw std::runtime_error(
            "Failed to generate regenerated checksum file");
      std::string diff_cmd =
          "diff -q \"" + TEST_DIR + "/orig.md5\" \"" + regen_md5 + "\"";
      if (std::system(diff_cmd.c_str()) != 0)
        throw std::runtime_error("Checksum files differ (orig vs regen)");
    }

    // --- Edge Case: remove_origin ---
    std::cout << "\n[Edge Case: remove_origin]" << std::endl;
    {
      if (cfg.verbosity >= 1)
        std::cout << "Testing compression with remove_origin flag..."
                  << std::endl;
      // Reset environment and clean zips
      setup_test_environment(true);
      TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                         TEST_VERBOSITY);
      // Configure remove_origin
      ConfigData cfg2 = cfg;
      cfg2.compress_mode = true;
      cfg2.remove_origin = true;
      auto items2 = FileHandler::discover_work_items({TEST_DIR}, cfg2);
      // Compress
      for (const auto &it : items2) {
        if (!Compressor::process_file(it.path, cfg2))
          throw std::runtime_error(
              "remove_origin test: compression failed for " + it.path);
      }
      // Verify originals deleted, zips exist
      for (const auto &pair : FILES_TO_CREATE) {
        const std::string orig = TEST_DIR + "/" + pair.first;
        const std::string zipf = orig + SUFFIX;
        if (std::filesystem::exists(orig))
          throw std::runtime_error(
              "remove_origin test: original still exists: " + orig);
        if (!std::filesystem::exists(zipf))
          throw std::runtime_error("remove_origin test: zip missing: " + zipf);
      }
      std::cout << "remove_origin edge case passed." << std::endl;
    }

    // --- Edge Case: decompress remove_origin ---
    std::cout << "\n[Edge Case: decompress remove_origin]" << std::endl;
    {
      // Setup and compress normally
      setup_test_environment(true);
      TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                         TEST_VERBOSITY);
      ConfigData cfg3 = cfg;
      cfg3.compress_mode = true;
      cfg3.remove_origin = false;
      auto items3 = FileHandler::discover_work_items({TEST_DIR}, cfg3);
      for (const auto &it : items3) {
        if (!Compressor::process_file(it.path, cfg3))
          throw std::runtime_error(
              "decompress_remove_origin setup: compression failed for " +
              it.path);
      }
      // Decompress with remove_origin=true
      cfg3.compress_mode = false;
      cfg3.remove_origin = true;
      auto items4 = FileHandler::discover_work_items({TEST_DIR}, cfg3);
      for (const auto &it : items4) {
        if (!Compressor::decompress_file(it.path, cfg3))
          throw std::runtime_error(
              "decompress_remove_origin test: decompression failed for " +
              it.path);
      }
      // Verify .zip removed and originals exist
      for (const auto &pair : FILES_TO_CREATE) {
        std::string zipf = TEST_DIR + "/" + pair.first + SUFFIX;
        std::string origf = TEST_DIR + "/" + pair.first;
        if (fs::exists(zipf))
          throw std::runtime_error(
              "decompress_remove_origin test: zip still exists: " + zipf);
        if (!fs::exists(origf))
          throw std::runtime_error(
              "decompress_remove_origin test: original missing: " + origf);
      }
      std::cout << "decompress_remove_origin edge case passed." << std::endl;
    }

    // --- Edge Case: invalid input path ---
    std::cout << "\n[Edge Case: invalid input path]" << std::endl;
    {
      std::string badpath = TEST_DIR + "/nonexistent.bin";
      bool caught = false;
      try {
        if (!Compressor::process_file(badpath, cfg))
          throw std::runtime_error("Expected failure on invalid path");
      } catch (...) {
        caught = true;
      }
      if (!caught)
        throw std::runtime_error(
            "invalid input path test: no exception for bad path");
      std::cout << "invalid input path edge case passed." << std::endl;
    }

    // --- Edge Case: threshold override ---
    std::cout << "\n[Edge Case: threshold override]" << std::endl;
    {
      setup_test_environment(true);
      TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                         TEST_VERBOSITY);
      ConfigData cfg4 = cfg;
      cfg4.compress_mode = true;
      cfg4.remove_origin = false;
      cfg4.large_file_threshold = 1; // Force all files to large path
      auto items5 = FileHandler::discover_work_items({TEST_DIR}, cfg4);
      for (const auto &it : items5) {
        if (!Compressor::process_file(it.path, cfg4))
          throw std::runtime_error(
              "threshold override test: compression failed for " + it.path);
      }
      // Decompress normally
      cfg4.compress_mode = false;
      auto items6 = FileHandler::discover_work_items({TEST_DIR}, cfg4);
      for (const auto &it : items6) {
        if (!Compressor::decompress_file(it.path, cfg4))
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

    // --- Edge Case: recursion disabled/enabled ---
    std::cout << "\n[Edge Case: recursion disabled/enabled]" << std::endl;
    {
      setup_test_environment(true);
      TestUtils::clean_files_with_suffix(TEST_DIR, SUFFIX, true,
                                         TEST_VERBOSITY);
      // Remove backup originals directory to avoid duplicate scanning
      fs::remove_all(ORIGINALS_DIR);
      // Config without recursion
      ConfigData cfg5 = cfg;
      cfg5.compress_mode = true;
      cfg5.remove_origin = false;
      cfg5.recurse = false;
      auto items_nr = FileHandler::discover_work_items({TEST_DIR}, cfg5);
      size_t expected_nr = FILES_TO_CREATE.size() - 1; // exclude subdir file
      if (items_nr.size() != expected_nr)
        throw std::runtime_error("recursion disabled test: expected " +
                                 std::to_string(expected_nr) + " items, got " +
                                 std::to_string(items_nr.size()));
      // Ensure subdir file not in list
      for (auto &it : items_nr) {
        if (it.path.find("subdir/") != std::string::npos)
          throw std::runtime_error(
              "recursion disabled test: found subdir item " + it.path);
      }
      // Config with recursion
      cfg5.recurse = true;
      auto items_r = FileHandler::discover_work_items({TEST_DIR}, cfg5);
      size_t expected_r = FILES_TO_CREATE.size();
      if (items_r.size() != expected_r)
        throw std::runtime_error("recursion enabled test: expected " +
                                 std::to_string(expected_r) + " items, got " +
                                 std::to_string(items_r.size()));
      std::cout << "recursion edge case passed." << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "!!! Test FAILED: " << e.what() << std::endl;
    all_passed = false;
  }

  cleanup_test_environment(); // Cleanup after tests for this mode

  // Finish banner
  std::cout << std::string(50, '=') << std::endl;
  std::cout << "  Tests Completed ("
            << (cfg.num_threads == 1 ? "Sequential" : "Parallel") << ")"
            << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  return all_passed;
}

// --- Main ---
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
