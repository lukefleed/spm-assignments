#ifndef MINIZP_TEST_UTILS_HPP
#define MINIZP_TEST_UTILS_HPP

#include <cstddef> // For size_t
#include <string>

namespace TestUtils {

/**
 * @brief Creates a file with pseudo-random binary content.
 * Uses /dev/urandom or similar platform-specific source if available,
 * otherwise falls back to C++ <random>.
 *
 * @param path The full path where the file should be created.
 * @param size The desired size of the file in bytes.
 * @param verbosity Verbosity level for error messages.
 * @return true on success, false on failure.
 */
bool create_random_file(const std::string &path, size_t size,
                        int verbosity = 1);

/**
 * @brief Compares two files byte-by-byte.
 *
 * @param path1 Path to the first file.
 * @param path2 Path to the second file.
 * @param verbosity Verbosity level for error messages.
 * @return true if the files are identical, false otherwise (or if an error
 * occurs).
 */
bool compare_files(const std::string &path1, const std::string &path2,
                   int verbosity = 1);

/**
 * @brief Cleans up (removes) files with a specific suffix in a directory.
 *
 * @param directory Path to the directory to clean.
 * @param suffix The suffix of files to remove (e.g., ".zip").
 * @param recursive If true, also cleans subdirectories.
 * @param verbosity Verbosity level for messages.
 * @return true if successful (or no files to remove), false on error.
 */
bool clean_files_with_suffix(const std::string &directory,
                             const std::string &suffix, bool recursive = false,
                             int verbosity = 1);

} // namespace TestUtils

#endif // MINIZP_TEST_UTILS_HPP
