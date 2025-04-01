#include "utils.h"
#include <iostream>
#include <optional> // For std::optional return type in parse_arguments
#include <sstream> // Potentially useful for string manipulation, though not used currently
#include <stdexcept> // For std::stoull exceptions (invalid_argument, out_of_range)
#include <string>    // Required for std::string operations
#include <vector>

/**
 * @brief Parses a string representation of a range (e.g., "start-end") into a
 * Range struct.
 * @param s The input string containing the range definition.
 * @param[out] range The Range struct to populate with the parsed start and end
 * values.
 * @return true if the string was successfully parsed into a valid range, false
 * otherwise.
 * @note Handles potential errors like invalid format, non-numeric values, or
 * out-of-range numbers. Also enforces that start <= end and start > 0
 * (adjusting start=0 to 1 with a warning).
 */
bool parse_range_string(const std::string &s, Range &range) {
  // Find the position of the delimiter '-'.
  size_t dash_pos = s.find('-');

  // Validate the delimiter's position: it must exist and not be at the very
  // start or end.
  if (dash_pos == std::string::npos || dash_pos == 0 ||
      dash_pos == s.length() - 1) {
    std::cerr << "Error: Invalid range format '" << s
              << "'. Expected 'start-end'." << std::endl;
    return false;
  }

  // Extract the substrings for start and end values.
  std::string start_str = s.substr(0, dash_pos);
  std::string end_str = s.substr(dash_pos + 1);

  try {
    // Helper function to parse numeric values with optional suffixes (k, M)
    auto parse_with_suffix = [](const std::string &num_str) -> ull {
      // Check if the string is empty
      if (num_str.empty()) {
        throw std::invalid_argument("Empty number string");
      }

      // Get the last character to check for suffix
      char last_char = num_str.back();
      std::string value_str = num_str;
      ull multiplier = 1;

      // Handle suffixes
      if (last_char == 'k' || last_char == 'K') {
        value_str = num_str.substr(0, num_str.length() - 1);
        multiplier = 1000; // thousand
      } else if (last_char == 'M') {
        value_str = num_str.substr(0, num_str.length() - 1);
        multiplier = 1000000; // million
      }

      // Convert to number and apply multiplier
      return std::stoull(value_str) * multiplier;
    };

    // Convert substrings to unsigned long long with suffix support
    range.start = parse_with_suffix(start_str);
    range.end = parse_with_suffix(end_str);

    // Handle the specific case of start being 0.
    if (range.start == 0) {
      std::cerr
          << "Warning: Range start is 0 in '" << s << "'. "
          << "Collatz is defined for positive integers. Adjusting start to 1."
          << std::endl;
      range.start = 1;
    }

    // Validate the logical order of the range.
    if (range.start > range.end) {
      std::cerr << "Warning: Range start (" << range.start
                << ") is greater than end (" << range.end << ") in '" << s
                << "'. Skipping this range." << std::endl;
      return false;
    }

    // If all conversions and checks pass, the range is valid.
    return true;

  } catch (const std::invalid_argument &e) {
    // Catch errors if the string does not represent a valid number.
    std::cerr << "Error: Invalid number format in range string '" << s << "'. "
              << e.what() << std::endl;
    return false;
  } catch (const std::out_of_range &e) {
    // Catch errors if the number is outside the representable range of unsigned
    // long long.
    std::cerr
        << "Error: Number out of range for unsigned long long in range string '"
        << s << "'. " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Prints the command-line usage instructions to standard error.
 * @param prog_name The name of the executable (argv[0]).
 */
void print_usage(const char *prog_name) {
  std::cerr << "Usage: " << prog_name << " [options] range1 [range2] ..."
            << std::endl;
  std::cerr << "  Calculates the maximum Collatz steps for numbers within "
               "specified ranges."
            << std::endl
            << std::endl;
  std::cerr << "Arguments:" << std::endl;
  std::cerr << "  rangeN        Required. One or more ranges in the format "
               "start-end (e.g., 1-1000)."
            << std::endl
            << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  -s <variant>  Static scheduling variant: block, cyclic, "
               "block-cyclic (default: block-cyclic)."
            << std::endl;
  std::cerr
      << "                (Effective only when -n > 1 and -d is not specified)."
      << std::endl;
  std::cerr << "  -d            Use dynamic work stealing scheduling."
            << std::endl;
  std::cerr << "                (Overrides static scheduling if both are "
               "specified implicitly or explicitly)."
            << std::endl;
  std::cerr << "  -n <threads>  Number of threads for parallel execution "
               "(default: 1, i.e., sequential)."
            << std::endl;
  std::cerr << "  -c <size>     Chunk size for dynamic scheduling or block "
               "size for static block-cyclic."
            << std::endl;
  std::cerr << "                Must be positive (default: 64)." << std::endl;
  // Removed default 1 to 64 for chunk size as it's often cache related.
  std::cerr
      << "  -v            Enable verbose output (prints execution details)."
      << std::endl;
  std::cerr << "  -t, --theory  Run theoretical analysis and generate speedup "
               "predictions."
            << std::endl;
  std::cerr << "                Results are saved to "
               "'results/theoretical_speedup.csv'."
            << std::endl;
  std::cerr << "  -h, --help    Show this help message and exit." << std::endl
            << std::endl;
  std::cerr << "Examples:" << std::endl;
  std::cerr << "  " << prog_name
            << " 1-10000               # Sequential execution" << std::endl;
  std::cerr << "  " << prog_name
            << " -n 8 1-1000000         # Static block-cyclic with 8 threads "
               "(default chunk size)"
            << std::endl;
  std::cerr << "  " << prog_name
            << " -d -n 16 -c 1024 1-1M  # Dynamic scheduling, 16 threads, "
               "chunk size 1024 (use M for million)"
            << std::endl;
  std::cerr << "  " << prog_name
            << " -s block -n 4 1-1k 10k-20k # Static block, 4 threads, "
               "multiple ranges (use k for thousand)"
            << std::endl;
  std::cerr << "  " << prog_name
            << " --theory               # Run theoretical analysis only"
            << std::endl;
}

/**
 * @brief Parses command-line arguments to configure the Collatz calculation.
 * @param argc The argument count from main.
 * @param argv The argument vector from main.
 * @return An std::optional<Config> containing the parsed configuration if
 * successful, or std::nullopt if parsing fails or help is requested.
 * @note Handles options for scheduling type, thread count, chunk size,
 * verbosity, and parses multiple range arguments. Sets reasonable defaults.
 */
std::optional<Config> parse_arguments(int argc, char *argv[]) {
  Config config; // Initialize with default values (see common_types.h/Config
                 // definition)
  std::vector<std::string>
      range_args; // Store positional arguments assumed to be ranges.

  // Iterate through command-line arguments, skipping the program name
  // (argv[0]).
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    // --- Option Parsing ---
    if (arg == "-d") {
      config.scheduling = SchedulingType::DYNAMIC;
    } else if (arg == "-s") {
      // The -s option requires an argument (the variant name).
      if (++i < argc) {
        std::string variant = argv[i];
        if (variant == "block") {
          config.static_variant = StaticVariant::BLOCK;
        } else if (variant == "cyclic") {
          config.static_variant = StaticVariant::CYCLIC;
        } else if (variant == "block-cyclic") {
          config.static_variant = StaticVariant::BLOCK_CYCLIC;
        } else {
          std::cerr << "Error: Unknown static scheduling variant '" << variant
                    << "'. Valid options: block, cyclic, block-cyclic."
                    << std::endl;
          print_usage(argv[0]);
          return std::nullopt; // Indicate parsing failure.
        }
        // Implicitly set scheduling type to STATIC if a variant is chosen,
        // unless -d was also specified (dynamic takes precedence if both
        // appear).
        if (config.scheduling != SchedulingType::DYNAMIC) {
          config.scheduling = SchedulingType::STATIC;
        }
      } else {
        std::cerr
            << "Error: Missing argument for static scheduling variant (-s)."
            << std::endl;
        print_usage(argv[0]);
        return std::nullopt;
      }
    } else if (arg == "-n") {
      // The -n option requires an argument (number of threads).
      if (++i < argc) {
        try {
          int threads = std::stoi(argv[i]); // Use stoi for integer conversion.
          if (threads <= 0) {
            std::cerr << "Error: Number of threads (-n) must be positive."
                      << std::endl;
            return std::nullopt;
          }
          config.num_threads = static_cast<unsigned int>(threads);
        } catch (const std::exception &e) {
          std::cerr << "Error: Invalid number specified for threads (-n): "
                    << argv[i] << ". " << e.what() << std::endl;
          return std::nullopt;
        }
      } else {
        std::cerr << "Error: Missing argument for number of threads (-n)."
                  << std::endl;
        print_usage(argv[0]);
        return std::nullopt;
      }
    } else if (arg == "-c") {
      // The -c option requires an argument (chunk/block size).
      if (++i < argc) {
        try {
          ull size = std::stoull(argv[i]); // Use stoull for unsigned long long.
          if (size == 0) {
            // Chunk size 0 is invalid for dynamic and block-cyclic strategies.
            std::cerr << "Error: Chunk/Block size (-c) must be positive."
                      << std::endl;
            return std::nullopt;
          }
          config.chunk_size = size;
        } catch (const std::exception &e) {
          std::cerr
              << "Error: Invalid number specified for chunk/block size (-c): "
              << argv[i] << ". " << e.what() << std::endl;
          return std::nullopt;
        }
      } else {
        std::cerr << "Error: Missing argument for chunk/block size (-c)."
                  << std::endl;
        print_usage(argv[0]);
        return std::nullopt;
      }
    } else if (arg == "-v") {
      config.verbose = true;
    } else if (arg == "-t" || arg == "--theory") {
      // config.theoretical_analysis = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      // Returning nullopt for help request signals the caller (main) not to
      // proceed with execution.
      return std::nullopt;
    } else if (arg[0] == '-') {
      // Handle unrecognized options starting with '-'.
      std::cerr << "Error: Unknown option '" << arg << "'." << std::endl;
      print_usage(argv[0]);
      return std::nullopt;
    } else {
      // --- Positional Argument Parsing ---
      // If it doesn't look like an option, assume it's a range argument.
      range_args.push_back(arg);
    }
  }

  // --- Validation after parsing all arguments ---

  // Ensure at least one range argument was provided.
  if (range_args.empty()) {
    std::cerr << "Error: No ranges specified." << std::endl;
    print_usage(argv[0]);
    return std::nullopt;
  }

  // Parse the collected range strings.
  for (const auto &r_str : range_args) {
    Range r;
    if (parse_range_string(r_str, r)) {
      config.ranges.push_back(r); // Add successfully parsed range.
    } else {
      // Error or warning was already printed by parse_range_string.
      // Decide whether to proceed or fail based on the specific error.
      // Currently, we only fail hard if *no* valid ranges can be parsed at all.
      // Warnings (like start > end) allow continuation if other ranges are
      // valid. If a fundamental parsing error occurred (format, non-numeric),
      // we might choose to fail immediately here, but the current approach is
      // more lenient. Check if the error was *not* the "start > end" warning
      // before printing another message. No need for extra error message here,
      // parse_range_string handles it.
    }
  }

  // After attempting to parse all range arguments, check if any were valid.
  if (config.ranges.empty()) {
    std::cerr
        << "Error: No valid ranges could be parsed from the provided arguments."
        << std::endl;
    // Usage might have been printed already if specific parse errors occurred.
    return std::nullopt;
  }

  // Final check: If dynamic scheduling is selected, chunk_size must be
  // positive. The default chunk_size is positive, so this mainly catches the
  // case where the user explicitly sets -c 0 with -d.
  if (config.scheduling == SchedulingType::DYNAMIC && config.chunk_size == 0) {
    std::cerr
        << "Error: Dynamic scheduling requires a positive chunk size (-c)."
        << std::endl;
    // Reset to default or keep the error? Let's enforce the rule.
    return std::nullopt;
    // Or: config.chunk_size = 64; std::cerr << "Warning: Resetting chunk size
    // to default 64 for dynamic scheduling." << std::endl;
  }

  // If num_threads is 1, force scheduling type to SEQUENTIAL for consistency.
  // This simplifies logic elsewhere, ensuring T=1 always means sequential
  // execution.
  if (config.num_threads == 1) {
    config.scheduling = SchedulingType::SEQUENTIAL;
    config.static_variant = StaticVariant::BLOCK; // Reset variant to default
    // Chunk size is ignored in sequential mode, no need to reset.
  } else if (config.scheduling == SchedulingType::SEQUENTIAL) {
    // If num_threads > 1 and no scheduling option was given, default to static
    config.scheduling = SchedulingType::STATIC;
  }

  // If all parsing and validation steps passed, return the populated Config
  return config;
}
