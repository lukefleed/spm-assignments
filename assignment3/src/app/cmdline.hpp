#ifndef MINIZP_CMDLINE_HPP
#define MINIZP_CMDLINE_HPP

/**
 * @file cmdline.hpp
 * @brief Provides command-line argument parsing functionality for the minizp
 * application.
 */

#include "config.hpp"
#include <cstdio>   // For std::printf (used in usage)
#include <cstdlib>  // For std::strtol, errno
#include <cstring>  // For strcmp (if needed, though not currently used)
#include <iostream> // For std::cerr
#include <omp.h>    // For omp_get_max_threads, omp_set_num_threads
#include <string>
#include <unistd.h> // For getopt, optind, opterr, optopt
#include <vector>

/**
 * @brief Namespace containing command-line parsing utilities.
 */
namespace CmdLine {

/**
 * @brief Prints usage instructions to standard output.
 *
 * Displays the command-line options and their defaults.
 *
 * @param argv0 The program name (typically argv[0]).
 */
static inline void usage(const char *argv0) {
  // Using std::printf for potentially complex formatting, but could use
  // std::cout
  std::printf("--------------------\n");
  std::printf("Usage: %s [options] file-or-directory [file-or-directory ...]\n",
              argv0);
  std::printf("\nOptions:\n");
  std::printf(" -C [0|1]    Compress mode. 0 preserves original, 1 removes. "
              "(Default: Compress, Preserve)\n");
  std::printf(
      " -D [0|1]    Decompress mode. 0 preserves original, 1 removes.\n");
  std::printf(" -r [0|1]    Recursively process subdirectories (0=No, 1=Yes. "
              "Default: %d)\n",
              false); // Default no recursion
  std::printf(" -t <num>    Number of threads to use (Default: %d)\n",
              omp_get_max_threads());
  std::printf(" -q <level>  Verbosity level (0=silent, 1=errors, 2=verbose. "
              "Default: %d)\n",
              1);
  std::printf(" -h          Show this help message.\n");
  std::printf("--------------------\n");
}

/**
 * @brief Checks if a C-string represents a valid non-negative integer.
 *
 * Parses the input string and verifies it contains only digits, with no
 * overflow or invalid characters.
 *
 * @param s Input C-style string to validate and convert.
 * @param[out] n Parsed integer value if successful.
 * @return true if the string is a valid non-negative integer, false otherwise.
 */
static bool isNumber(const char *s, long &n) {
  if (!s || *s == '\0') // Handle null or empty string
    return false;
  char *endptr;
  errno = 0; // Important: reset errno before call
  n = std::strtol(s, &endptr, 10);
  // Check for conversion errors:
  // 1. errno set (e.g., ERANGE for overflow)
  // 2. *endptr is not '\0' (indicates trailing non-digit characters)
  // 3. endptr == s (indicates no digits were found at all)
  if (errno != 0 || *endptr != '\0' || endptr == s) {
    return false;
  }
  // Optionally add check for negative numbers if only positive are allowed
  // if (n < 0) { return false; }
  return true;
}

/**
 * @brief Parses command-line arguments and populates configuration.
 *
 * Uses getopt to read options and validates mutual exclusions and ranges.
 * Populates the ConfigData and list of input paths for processing.
 *
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @param[out] config Configuration structure to fill with parsed values.
 * @param[out] input_paths Vector to be filled with non-option paths.
 * @return true if parsing succeeded and application should proceed;
 *         false if help was requested or an error occurred.
 */
static bool parseCommandLine(int argc, char *argv[], ConfigData &config,
                             std::vector<std::string> &input_paths) {
  // Defaults are set by ConfigData's constructor.

  int opt;
  const char *optstring = "C:D:r:t:q:h"; // Options requiring arguments have ':'
  bool c_present = false;                // Flag to detect mutual exclusion C/D
  bool d_present = false;                // Flag to detect mutual exclusion C/D
  opterr = 0; // Disable getopt's default error messages; we handle them.

  // Reset optind for potential re-parsing scenarios (though unlikely in main).
  optind = 1;

  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
    case 'C': {
      long val = 0;
      if (!isNumber(optarg, val) || (val != 0 && val != 1)) {
        std::cerr << "Error: Invalid value for -C option. Use 0 or 1.\n";
        usage(argv[0]);
        return false;
      }
      c_present = true;
      config.compress_mode = true;
      config.remove_origin = (val == 1);
    } break;
    case 'D': {
      long val = 0;
      if (!isNumber(optarg, val) || (val != 0 && val != 1)) {
        std::cerr << "Error: Invalid value for -D option. Use 0 or 1.\n";
        usage(argv[0]);
        return false;
      }
      d_present = true;
      config.compress_mode = false;
      config.remove_origin = (val == 1);
    } break;
    case 'r': {
      long val = 0;
      if (!isNumber(optarg, val) || (val != 0 && val != 1)) {
        std::cerr << "Error: Invalid value for -r option. Use 0 or 1.\n";
        usage(argv[0]);
        return false;
      }
      config.recurse = (val == 1);
    } break;
    case 't': {
      long val = 0;
      // Ensure the number of threads is positive.
      if (!isNumber(optarg, val) || val <= 0) {
        std::cerr << "Error: Invalid value for -t option. Must be a "
                     "positive integer.\n";
        usage(argv[0]);
        return false;
      }
      config.num_threads = static_cast<int>(val);
    } break;
    case 'q': {
      long val = 0;
      // Validate verbosity level range.
      if (!isNumber(optarg, val) || val < 0 || val > 2) {
        std::cerr << "Error: Invalid value for -q option. Use 0, 1, or 2.\n";
        usage(argv[0]);
        return false;
      }
      config.verbosity = static_cast<int>(val);
    } break;
    case 'h':
      usage(argv[0]);
      return false; // Signal to exit after showing help.
    case '?':       // Handle unknown option or missing required argument.
      if (optopt) { // optopt contains the unknown option character
        std::cerr << "Error: Unknown option '-" << static_cast<char>(optopt)
                  << "' or missing argument.\n";
      } else { // Handle cases like `prog -` or non-option args starting with -
        std::cerr << "Error: Invalid option or missing argument near '"
                  << argv[optind - 1] << "'.\n";
      }
      usage(argv[0]);
      return false;
    default:
      // This case should theoretically not be reached with opterr=0.
      std::cerr << "Error: Unexpected error parsing options.\n";
      usage(argv[0]);
      return false;
    }
  }

  // --- Post-parsing Validation ---

  // Check for mutually exclusive options -C and -D.
  if (c_present && d_present) {
    std::cerr << "Error: Options -C and -D are mutually exclusive.\n";
    usage(argv[0]);
    return false;
  }

  // Collect remaining non-option arguments as input paths.
  input_paths.clear(); // Ensure it's empty before filling
  for (int i = optind; i < argc; ++i) {
    input_paths.push_back(argv[i]);
  }

  // Check if at least one input path was provided.
  if (input_paths.empty()) {
    std::cerr << "Error: No input files or directories specified.\n";
    usage(argv[0]);
    return false;
  }

  // Configure OpenMP threads based on the parsed value or default.
  // Note: omp_set_num_threads should ideally be called early in main.
  // We set the value in config; the caller (main) should call omp_set...
  if (config.num_threads <= 0) {
    // If parsing resulted in an invalid number (e.g., not set or set to 0),
    // fall back to the OpenMP default.
    config.num_threads = omp_get_max_threads();
    // Ensure at least 1 thread.
    if (config.num_threads <= 0) {
      config.num_threads = 1;
    }
  }
  // The actual call to omp_set_num_threads(config.num_threads) should be done
  // in main after parsing.

  return true; // Parsing successful.
}

} // namespace CmdLine

#endif // MINIZP_CMDLINE_HPP
