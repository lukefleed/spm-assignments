/**
 * @file compressor.hpp
 * @brief Declarations for file compression and decompression routines,
 *        supporting small and block-based large file processing.
 */
#ifndef MINIZP_COMPRESSOR_HPP
#define MINIZP_COMPRESSOR_HPP

#include "config.hpp" /**< Shared configuration data and constants */
#include <cstdint>    /**< For fixed-width integer types */
#include <string>
#include <vector>

namespace Compressor {
using ::FORMAT_VERSION;
using ::MAGIC_NUMBER_LARGE_FILE;
using ::SUFFIX;

/**
 * @brief Header structure for custom large compressed files.
 * Written at the beginning of the .zip file.
 */
#pragma pack(push, 1) // Ensure struct is packed tightly
struct LargeFileHeader {
  uint32_t magic_number = MAGIC_NUMBER_LARGE_FILE;
  uint16_t version = FORMAT_VERSION;
  uint64_t original_size = 0;
  uint64_t num_blocks = 0;
  // Block metadata (compressed sizes) will follow this header directly in the
  // file.
};
#pragma pack(pop)

/**
 * @brief Processes (compresses) a single file.
 *
 * Determines if the file is small or large based on config.large_file_threshold
 * and calls the appropriate compression function (compress_small_file or
 * compress_large_file).
 *
 * @param input_path Path to the input file.
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool process_file(const std::string &input_path, const ConfigData &cfg);

/**
 * @brief Decompresses a single file.
 *
 * Reads the beginning of the file to check for the custom large file header
 * magic number. Calls the appropriate decompression function
 * (decompress_small_file or decompress_large_file).
 *
 * @param input_path Path to the compressed input file (expected .zip suffix).
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool decompress_file(const std::string &input_path, const ConfigData &cfg);

// --- Potentially internal functions (could be in .cpp in anonymous namespace)
// ---

/**
 * @brief Compresses a "small" file (below threshold) using standard Miniz
 * compression. Creates a standard .zip file (though technically just zlib
 * stream, not zip archive).
 *
 * @param input_path Path to the input file.
 * @param input_size Size of the input file.
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool compress_small_file(const std::string &input_path, size_t input_size,
                         const ConfigData &cfg);

/**
 * @brief Compresses a "large" file (at or above threshold) by splitting into
 * blocks and compressing them in parallel using OpenMP. Creates a .zip file
 * with a custom header format.
 *
 * @param input_path Path to the input file.
 * @param input_size Size of the input file.
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool compress_large_file(const std::string &input_path, size_t input_size,
                         const ConfigData &cfg);

/**
 * @brief Decompresses a "small" file (standard zlib stream).
 *
 * @param input_path Path to the compressed input file.
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool decompress_small_file(const std::string &input_path,
                           const ConfigData &cfg);

/**
 * @brief Decompresses a "large" file created with the custom block format.
 * Reads the header, allocates memory, and decompresses blocks (sequentially
 * initially).
 *
 * @param input_path Path to the compressed input file.
 * @param cfg Configuration data.
 * @return true on success, false on failure.
 */
bool decompress_large_file(const std::string &input_path,
                           const ConfigData &cfg);

} // namespace Compressor

#endif // MINIZP_COMPRESSOR_HPP
