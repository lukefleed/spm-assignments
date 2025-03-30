#include "utils.h"
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept> // Per std::stoull
#include <vector>

bool parse_range_string(const std::string &s, Range &range) {
  size_t dash_pos = s.find('-');
  if (dash_pos == std::string::npos || dash_pos == 0 ||
      dash_pos == s.length() - 1) {
    return false; // Formato non valido
  }
  std::string start_str = s.substr(0, dash_pos);
  std::string end_str = s.substr(dash_pos + 1);

  try {
    // Usare stoull per unsigned long long
    range.start = std::stoull(start_str);
    range.end = std::stoull(end_str);

    if (range.start == 0) {
      std::cerr << "Warning: Range start is 0. Collatz is defined for positive "
                   "integers. Starting calculation from 1."
                << std::endl;
      range.start = 1; // Correggiamo o decidiamo come gestire 0
    }

    if (range.start > range.end) {
      std::cerr << "Warning: Range start (" << range.start
                << ") is greater than end (" << range.end
                << "). Skipping this range." << std::endl;
      return false; // Intervallo non valido
    }
    return true;
  } catch (const std::invalid_argument &e) {
    std::cerr << "Error parsing range number: " << e.what() << " in string '"
              << s << "'" << std::endl;
    return false;
  } catch (const std::out_of_range &e) {
    std::cerr << "Error: Range number out of range for unsigned long long: "
              << e.what() << " in string '" << s << "'" << std::endl;
    return false;
  }
  return false; // Non dovrebbe arrivare qui
}

void print_usage(const char *prog_name) {
  std::cerr << "Usage: " << prog_name << " [options] range1 [range2] ..."
            << std::endl;
  std::cerr << "  range format: start-end (e.g., 1-1000)" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr
      << "  -d            Use dynamic scheduling (default: static block-cyclic)"
      << std::endl;
  std::cerr << "  -n <threads>  Number of threads (default: 16)" << std::endl;
  std::cerr << "  -c <size>     Block/Chunk size (default: 1)" << std::endl;
  std::cerr << "  -s <variant>  Static scheduling variant (block, cyclic, "
               "block-cyclic)"
            << std::endl;
  std::cerr << "                Only relevant when using static scheduling"
            << std::endl;
  std::cerr << "  -h, --help    Show this help message" << std::endl;
  std::cerr << "  -v            Verbose output (for debugging)" << std::endl;
}

std::optional<Config> parse_arguments(int argc, char *argv[]) {
  Config config;
  std::vector<std::string> range_args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-d") {
      config.scheduling = SchedulingType::DYNAMIC;
    } else if (arg == "-s") {
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
                    << "'. Valid options: block, cyclic, block-cyclic"
                    << std::endl;
          return std::nullopt;
        }
      } else {
        std::cerr << "Error: Missing argument for -s" << std::endl;
        return std::nullopt;
      }
    } else if (arg == "-n") {
      if (++i < argc) {
        try {
          config.num_threads = std::stoi(argv[i]);
          if (config.num_threads <= 0) {
            std::cerr << "Error: Number of threads must be positive."
                      << std::endl;
            return std::nullopt;
          }
        } catch (const std::exception &e) {
          std::cerr << "Error parsing number of threads: " << e.what()
                    << std::endl;
          return std::nullopt;
        }
      } else {
        std::cerr << "Error: Missing argument for -n" << std::endl;
        return std::nullopt;
      }
    } else if (arg == "-c") {
      if (++i < argc) {
        try {
          config.chunk_size = std::stoull(argv[i]);
          if (config.chunk_size == 0) {
            std::cerr << "Error: Chunk/Block size must be positive."
                      << std::endl;
            return std::nullopt;
          }
        } catch (const std::exception &e) {
          std::cerr << "Error parsing chunk size: " << e.what() << std::endl;
          return std::nullopt;
        }
      } else {
        std::cerr << "Error: Missing argument for -c" << std::endl;
        return std::nullopt;
      }
    } else if (arg == "-v") {
      config.verbose = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return std::nullopt; // Non è un errore, ma non dobbiamo eseguire il
                           // calcolo
    } else if (arg[0] == '-') {
      std::cerr << "Error: Unknown option '" << arg << "'" << std::endl;
      print_usage(argv[0]);
      return std::nullopt;
    } else {
      // Assume che sia un argomento range
      range_args.push_back(arg);
    }
  }

  if (range_args.empty()) {
    std::cerr << "Error: No ranges provided." << std::endl;
    print_usage(argv[0]);
    return std::nullopt;
  }

  for (const auto &r_str : range_args) {
    Range r;
    if (parse_range_string(r_str, r)) {
      config.ranges.push_back(r);
    } else {
      // Errore già stampato da parse_range_string (se non warning di range
      // inverso)
      if (r.start <= r.end) { // Solo se l'errore non era start > end
        std::cerr << "Failed to parse range: " << r_str << std::endl;
      }
      // Decidiamo se continuare ignorando il range errato o fermarci
      // Per ora, continuiamo se ci sono altri range validi, altrimenti errore.
      // Se questo era l'unico range e fallisce, lo gestiamo dopo il loop.
    }
  }

  if (config.ranges.empty()) {
    std::cerr << "Error: No valid ranges could be parsed." << std::endl;
    // print_usage(argv[0]); // Già stampato se c'erano errori o nessun range
    return std::nullopt;
  }

  return config;
}
