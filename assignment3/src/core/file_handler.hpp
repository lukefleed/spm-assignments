#ifndef MINIZP_FILE_HANDLER_HPP
#define MINIZP_FILE_HANDLER_HPP

#include "config.hpp" // Uses ConfigData
#include <filesystem> // Requires C++17
#include <string>
#include <vector>

namespace FileHandler {

/**
 * @brief Represents an item (file or directory) discovered for processing.
 */
struct WorkItem {
  std::string path;          /**< Full path to the item. */
  size_t size = 0;           /**< Size in bytes (if it's a file). */
  bool is_directory = false; /**< True if the item is a directory. */
  // We might not need is_directory if discover_work_items only returns files
};

/**
 * @brief Checks if a path corresponds to a directory or a regular file.
 *
 * @param p The path to check.
 * @param filesize Output parameter: if it's a regular file, its size is stored
 * here.
 * @param verbosity Verbosity level for error messages.
 * @return true if it's a directory, false if it's a regular file.
 * @throws std::filesystem::filesystem_error on file system access errors
 * (unless handled internally).
 */
bool is_directory_or_file(const std::filesystem::path &p, size_t &filesize,
                          int verbosity);

/**
 * @brief Determines if a given filename should be processed based on mode and
 * suffix.
 *
 * @param filename The name of the file (not the full path).
 * @param is_compress_mode True if in compression mode, false for decompression.
 * @param suffix The suffix to check for (e.g., ".zip").
 * @return true if the file should be processed, false if it should be skipped.
 */
bool should_process(const std::string &filename, bool is_compress_mode,
                    const std::string &suffix);

/**
 * @brief Discovers all files to be processed based on initial paths and
 * configuration.
 *
 * Handles recursion and filtering based on the configuration.
 *
 * @param initial_paths Vector of starting file or directory paths from the
 * command line.
 * @param cfg The application configuration.
 * @return std::vector<WorkItem> A list of files (WorkItem entries) to be
 * processed. Directories are not included in the final list.
 * @throws std::filesystem::filesystem_error on file system access errors.
 */
std::vector<WorkItem>
discover_work_items(const std::vector<std::string> &initial_paths,
                    const ConfigData &cfg);

} // namespace FileHandler

#endif // MINIZP_FILE_HANDLER_HPP
