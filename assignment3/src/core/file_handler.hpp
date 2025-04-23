/**
 * @file file_handler.hpp
 * @brief Defines functions for discovering and filtering files based on
 * configuration. Handles path checking, recursion, and suffix filtering.
 */
#ifndef MINIZP_FILE_HANDLER_HPP
#define MINIZP_FILE_HANDLER_HPP

#include "config.hpp" // Uses ConfigData
#include <filesystem> // Requires C++17
#include <optional>
#include <string>
#include <vector>

namespace FileHandler {

using ::FORMAT_VERSION;
using ::MAGIC_NUMBER_LARGE_FILE;
using ::SUFFIX;

/**
 * @brief Represents a file discovered for processing.
 */
struct WorkItem {
  std::string path; /**< Full path to the file. */
  size_t size = 0;  /**< Size in bytes. */
  // is_directory field removed: discover_work_items only returns files.
};

/**
 * @brief Checks if a path corresponds to a directory.
 * @param[in] p The path to check.
 * @param[in] verbosity Verbosity level for logging errors.
 * @return true if the path is a directory, false otherwise.
 * @throws std::filesystem::filesystem_error on underlying file system errors if
 * not handled internally based on verbosity.
 */
bool is_directory(const std::filesystem::path &p, int verbosity);

/**
 * @brief Gets the size of a regular file.
 * @param[in] p The path to check.
 * @param[in] verbosity Verbosity level for logging errors.
 * @return std::optional<size_t> containing the file size if it's a regular file
 * and size could be obtained, std::nullopt otherwise.
 * @throws std::filesystem::filesystem_error on underlying file system errors if
 * not handled internally based on verbosity.
 */
std::optional<size_t> get_regular_file_size(const std::filesystem::path &p,
                                            int verbosity);

/**
 * @brief Determines if a given filename should be processed based on mode and
 * suffix.
 * @param[in] filename The name of the file (not the full path).
 * @param[in] is_compress_mode True if in compression mode, false for
 * decompression.
 * @param[in] suffix The suffix to check for (e.g., ".zip").
 * @return true if the file should be processed, false if it should be skipped.
 */
bool should_process(const std::string &filename, bool is_compress_mode,
                    const std::string &suffix);

/**
 * @brief Discovers all files to be processed based on initial paths and
 * configuration. Handles recursion and filtering based on the configuration.
 * @param[in] initial_paths Vector of starting file or directory paths from the
 * command line.
 * @param[in] cfg The application configuration.
 * @return std::vector<WorkItem> A list of files (WorkItem entries) to be
 * processed.
 * @throws std::filesystem::filesystem_error on file system access errors during
 * iteration if not handled internally.
 */
std::vector<WorkItem>
discover_work_items(const std::vector<std::string> &initial_paths,
                    const ConfigData &cfg);

} // namespace FileHandler

#endif // MINIZP_FILE_HANDLER_HPP
