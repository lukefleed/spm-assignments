#ifndef MINIZP_CONFIG_HPP
#define MINIZP_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <omp.h> // For omp_get_max_threads
#include <string>

// --- Constants ---
/**
 * @brief Suffix for compressed files (e.g., ".zip").
 */
inline const std::string SUFFIX = ".zip";

/**
 * @brief File size threshold to trigger large file processing logic. Default is
 * 16 MiB.
 */
// constexpr size_t LARGE_FILE_THRESHOLD_DEFAULT = 128 * 1024 * 1024; // 128 MiB
constexpr size_t LARGE_FILE_THRESHOLD_DEFAULT = 16 * 1024 * 1024; // 16 MiB

/**
 * @brief Size of blocks for large file processing. Default is 1 MiB.
 */
constexpr size_t BLOCK_SIZE_DEFAULT = 1 * 1024 * 1024; // 1 MiB

/**
 * @brief Magic number for large file format (ASCII for "MPBL").
 */
constexpr uint32_t MAGIC_NUMBER_LARGE_FILE =
    0x4D50424C; // ASCII for "MPBL" (Miniz Parallel Block)

/**
 * @brief Version number for large file format.
 */
constexpr uint16_t FORMAT_VERSION = 1;

// --- Configuration Structure ---
struct ConfigData {
  /** @brief Operation mode: true for compression, false for decompression. */
  bool compress_mode = true;

  /** @brief Whether to remove the original file after processing. */
  bool remove_origin = false;

  /** @brief Verbosity level: 0=silent, 1=errors only, 2=verbose info. */
  int verbosity = 1;

  /** @brief Whether to recurse into subdirectories. */
  bool recurse = false;

  /** @brief Number of OpenMP threads to use. Defaults to max available. */
  int num_threads = omp_get_max_threads();

  /** @brief File size threshold to trigger large file processing logic. */
  size_t large_file_threshold = LARGE_FILE_THRESHOLD_DEFAULT;

  /** @brief Size of blocks for large file processing. */
  size_t block_size = BLOCK_SIZE_DEFAULT;
};

#endif // MINIZP_CONFIG_HPP
