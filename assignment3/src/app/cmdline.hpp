#ifndef MINIZP_CMDLINE_HPP
#define MINIZP_CMDLINE_HPP

#include "config.hpp" // Include first to get ConfigData definition
#include <cstdio>     // For fprintf, stderr
#include <cstdlib>    // For stol, strtol error handling
#include <string>
#include <unistd.h> // For getopt
#include <vector>

namespace CmdLine {

/**
 * @brief Prints usage instructions to stdout.
 * @param argv0 The name of the executable (argv[0]).
 */
static inline void usage(const char *argv0) {
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
  // Add options for threshold and block size if desired
  // std::printf(" --threshold <bytes>  Threshold for large file processing
  // (Default: %zu)\n", LARGE_FILE_THRESHOLD_DEFAULT); std::printf(" --blocksize
  // <bytes>  Block size for large file processing (Default: %zu)\n",
  // BLOCK_SIZE_DEFAULT);
  std::printf(" -h          Show this help message.\n");
  std::printf("--------------------\n");
}

/**
 * @brief Checks if a string represents a valid number and converts it.
 * @param s The input string.
 * @param n Output parameter for the converted number.
 * @return true if the string is a valid number, false otherwise.
 */
static bool isNumber(const char *s, long &n) {
  if (!s)
    return false;
  char *endptr;
  errno = 0; // Important: reset errno before call
  n = std::strtol(s, &endptr, 10);
  // Check for conversion errors
  if (errno != 0 || *endptr != '\0' || endptr == s) {
    return false;
  }
  return true;
}

/**
 * @brief Parses command line arguments and populates configuration and input
 * paths.
 *
 * @param argc Argument count from main.
 * @param argv Argument vector from main.
 * @param config Output ConfigData structure to populate.
 * @param input_paths Output vector of initial file/directory paths provided.
 * @return true if parsing was successful and the program should continue, false
 * otherwise.
 */
static bool parseCommandLine(int argc, char *argv[], ConfigData &config,
                             std::vector<std::string> &input_paths) {
  // Set defaults based on ConfigData struct defaults
  // These are set during ConfigData initialization

  int opt;
  const char *optstring = "C:D:r:t:q:h"; // Added 't' and 'h'
  bool c_present = false;
  bool d_present = false;
  opterr = 0; // Disable default getopt error messages

  // Reset optind for potential multiple calls (though usually not needed in
  // main)
  optind = 1;

  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
    case 'C': {
      long val = 0;
      if (!isNumber(optarg, val) || (val != 0 && val != 1)) {
        std::fprintf(stderr,
                     "Error: Invalid value for -C option. Use 0 or 1.\n");
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
        std::fprintf(stderr,
                     "Error: Invalid value for -D option. Use 0 or 1.\n");
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
        std::fprintf(stderr,
                     "Error: Invalid value for -r option. Use 0 or 1.\n");
        usage(argv[0]);
        return false;
      }
      config.recurse = (val == 1);
    } break;
    case 't': {
      long val = 0;
      if (!isNumber(optarg, val) || val <= 0) {
        std::fprintf(stderr, "Error: Invalid value for -t option. Must be a "
                             "positive integer.\n");
        usage(argv[0]);
        return false;
      }
      config.num_threads = static_cast<int>(val);
    } break;
    case 'q': {
      long val = 0;
      if (!isNumber(optarg, val) || val < 0 || val > 2) {
        std::fprintf(stderr,
                     "Error: Invalid value for -q option. Use 0, 1, or 2.\n");
        usage(argv[0]);
        return false;
      }
      config.verbosity = static_cast<int>(val);
    } break;
    case 'h':
      usage(argv[0]);
      return false; // Exit after showing help
    case '?':       // Unknown option or missing argument
      std::fprintf(stderr, "Error: Unknown option '-%c' or missing argument.\n",
                   optopt);
      usage(argv[0]);
      return false;
    default:
      // Should not happen with opterr=0
      usage(argv[0]);
      return false;
    }
  }

  // --- Post-parsing checks ---
  if (c_present && d_present) {
    std::fprintf(stderr, "Error: Options -C and -D are mutually exclusive.\n");
    usage(argv[0]);
    return false;
  }

  // Collect remaining arguments as input paths
  for (int i = optind; i < argc; ++i) {
    input_paths.push_back(argv[i]);
  }

  if (input_paths.empty()) {
    std::fprintf(stderr, "Error: No input files or directories specified.\n");
    usage(argv[0]);
    return false;
  }

  // Set OpenMP threads if specified
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  } else {
    // Use default (all available cores) - already set in ConfigData
    config.num_threads =
        omp_get_max_threads(); // Ensure config reflects actual number
  }

  return true; // Parsing successful
}

} // namespace CmdLine

#endif // MINIZP_CMDLINE_HPP
