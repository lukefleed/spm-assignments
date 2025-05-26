#include "utils.h"
#include <algorithm> // For std::sort (for reference sort), std::is_sorted
#include <fstream>
#include <iomanip> // For std::setw, std::fixed, std::setprecision (in print_records_sample)
#include <iostream>
#include <numeric> // For std::accumulate (for checksum)
#include <random>  // For std::mt19937_64, std::uniform_int_distribution
#include <stdexcept> // For std::stoul, std::stoi, std::invalid_argument, std::out_of_range

// Parses command line arguments.
bool parse_arguments(int argc, char *argv[], Arguments &args) {
  // Default seed to a common value for reproducibility if not specified.
  // Time-based seed could be used for more varied runs: std::time(0)
  args.random_seed = 12345;

  for (int i = 1; i < argc; ++i) {
    std::string arg_str = argv[i];
    try {
      if (arg_str == "-s" && i + 1 < argc) {
        std::string n_val_str = argv[++i];
        char suffix = n_val_str.back();
        if (std::isalpha(suffix)) { // Check if last char is alphabetic
          n_val_str.pop_back();     // Remove suffix
          if (toupper(suffix) == 'M') {
            args.N_elements = std::stoull(n_val_str) * 1000000;
          } else if (toupper(suffix) == 'K') {
            args.N_elements = std::stoull(n_val_str) * 1000;
          } else {
            std::cerr << "Error: Invalid suffix for array size: " << suffix
                      << std::endl;
            return false;
          }
        } else {
          args.N_elements = std::stoull(n_val_str);
        }
      } else if (arg_str == "-r" && i + 1 < argc) {
        args.R_payload_size_bytes = std::stoul(argv[++i]);
        if (args.R_payload_size_bytes > MAX_RPAYLOAD_SIZE) {
          std::cerr << "Error: Requested payload size "
                    << args.R_payload_size_bytes
                    << " exceeds MAX_RPAYLOAD_SIZE " << MAX_RPAYLOAD_SIZE
                    << std::endl;
          return false;
        }
      } else if (arg_str == "-t" && i + 1 < argc) {
        args.T_ff_threads = std::stoi(argv[++i]);
      } else if (arg_str == "--input" && i + 1 < argc) {
        args.input_file_path = argv[++i];
      } else if (arg_str == "--output" && i + 1 < argc) {
        args.output_file_path = argv[++i];
      } else if (arg_str == "--check_correctness") {
        args.check_correctness = true;
      } else if (arg_str == "--perf_mode") {
        args.perf_mode = true;
      } else if (arg_str == "--seed" && i + 1 < argc) {
        args.random_seed = std::stoul(argv[++i]);
      } else {
        std::cerr << "Error: Unknown or incomplete argument: " << arg_str
                  << std::endl;
        return false;
      }
    } catch (const std::invalid_argument &ia) {
      std::cerr << "Error: Invalid numeric value for argument " << arg_str
                << ": " << argv[i] << std::endl;
      return false;
    } catch (const std::out_of_range &oor) {
      std::cerr << "Error: Numeric value out of range for argument " << arg_str
                << ": " << argv[i] << std::endl;
      return false;
    }
  }

  // Basic validation
  if (args.N_elements == 0 ||
      args.R_payload_size_bytes ==
          0) { // T_ff_threads can be 0 if not applicable e.g. sequential run
    std::cerr
        << "Usage: " << argv[0]
        << " -s N -r R [-t T_FF] [--input <filepath>] [--output <filepath>] "
           "[--check_correctness] [--perf_mode] [--seed <uint>]"
        << std::endl;
    std::cerr << "  Required:" << std::endl;
    std::cerr << "    -s N: array size (e.g., 10M, 100K, or 1000000)"
              << std::endl;
    std::cerr << "    -r R: record payload size in bytes (e.g., 8, 64, 256)"
              << std::endl;
    std::cerr << "  Optional for FastFlow/Hybrid versions:" << std::endl;
    std::cerr << "    -t T_FF: number of FastFlow threads (e.g., 4, 8)"
              << std::endl;
    std::cerr << "  Optional general:" << std::endl;
    std::cerr
        << "    --input <filepath>: Path to pre-generated binary input file."
        << std::endl;
    std::cerr << "    --output <filepath>: Path to save the sorted binary "
                 "output file."
              << std::endl;
    std::cerr << "    --check_correctness: Enable correctness checks against a "
                 "reference sort."
              << std::endl;
    std::cerr
        << "    --perf_mode: Minimal output, suitable for benchmark scripting."
        << std::endl;
    std::cerr << "    --seed <uint>: Seed for random data generation."
              << std::endl;
    return false;
  }
  // T_ff_threads is only strictly required if it's a parallel version that uses
  // it. The main_xx.cpp files can enforce this if -t is missing for them.
  return true;
}

// Generates random records.
void generate_random_records(Record *records_array, const Arguments &args) {
  std::mt19937_64 rng(args.random_seed); // 64-bit Mersenne Twister engine
  std::uniform_int_distribution<unsigned long>
      key_dist; // Full range for unsigned long
  std::uniform_int_distribution<int> payload_char_dist(
      0, 255); // For payload bytes

  for (size_t i = 0; i < args.N_elements; ++i) {
    Record *rec = &records_array[i]; // USE ARRAY INDEXING
    rec->key = key_dist(rng);
    for (size_t j = 0; j < args.R_payload_size_bytes; ++j) {
      rec->rpayload[j] = static_cast<char>(payload_char_dist(rng));
    }
    // Pad remaining MAX_RPAYLOAD_SIZE - R_payload_size_bytes with 0s
    for (size_t j = args.R_payload_size_bytes; j < MAX_RPAYLOAD_SIZE; ++j) {
      rec->rpayload[j] = 0;
    }
  }
}

// Loads records from a binary file.
bool load_records_from_file(Record *records_array, const Arguments &args) {
  if (args.input_file_path.empty()) {
    std::cerr << "Error: Input file path not provided for loading records."
              << std::endl;
    return false;
  }

  std::ifstream infile(args.input_file_path, std::ios::binary);
  if (!infile) {
    std::cerr << "Error: Cannot open input file: " << args.input_file_path
              << std::endl;
    return false;
  }

  for (size_t i = 0; i < args.N_elements; ++i) {
    Record *rec = &records_array[i];

    // Read key
    infile.read(reinterpret_cast<char *>(&rec->key), sizeof(unsigned long));
    if (!infile || infile.gcount() !=
                       static_cast<std::streamsize>(sizeof(unsigned long))) {
      std::cerr << "Error: Failed to read key for record " << i;
      if (infile.eof()) {
        std::cerr << " (unexpected EOF).";
      }
      std::cerr << " Expected " << args.N_elements << " records, read " << i
                << "." << std::endl;
      return false;
    }

    // Read only the actual payload size
    infile.read(rec->rpayload, args.R_payload_size_bytes);
    if (!infile || infile.gcount() != static_cast<std::streamsize>(
                                          args.R_payload_size_bytes)) {
      std::cerr << "Error: Failed to read payload for record " << i;
      if (infile.eof()) {
        std::cerr << " (unexpected EOF).";
      }
      std::cerr << " Expected " << args.N_elements << " records, read " << i
                << " (plus current key)." << std::endl;
      return false;
    }

    // Zero out the rest of MAX_RPAYLOAD_SIZE
    for (size_t j = args.R_payload_size_bytes; j < MAX_RPAYLOAD_SIZE; ++j) {
      rec->rpayload[j] = 0;
    }
  }

  // Check for extra data
  infile.peek();
  if (!infile.eof()) {
    std::cerr
        << "Warning: Input file contains more data than N_elements specified."
        << std::endl;
  }

  return true;
}

// Saves records to a binary file.
bool save_records_to_file(const Record *records_array, const Arguments &args) {
  if (args.output_file_path.empty()) {
    std::cerr << "Error: Output file path not provided for saving records."
              << std::endl;
    return false;
  }

  std::ofstream outfile(args.output_file_path,
                        std::ios::binary | std::ios::trunc);
  if (!outfile) {
    std::cerr << "Error: Cannot open output file: " << args.output_file_path
              << std::endl;
    return false;
  }

  for (size_t i = 0; i < args.N_elements; ++i) {
    const Record *rec = &records_array[i]; // USE ARRAY INDEXING
    // Write key
    outfile.write(reinterpret_cast<const char *>(&rec->key),
                  sizeof(unsigned long));
    // Write only the actual payload size
    outfile.write(rec->rpayload, args.R_payload_size_bytes);

    if (outfile.fail()) {
      std::cerr << "Error: Failed to write record " << i
                << " to file: " << args.output_file_path << std::endl;
      return false;
    }
  }
  return true;
}

// Verifies if the array of records is sorted.
// For a more robust check, original_records_array_copy can be a copy of the
// input array BEFORE sorting, to check for element loss/duplication via
// checksums. Currently, it primarily checks non-decreasing key order.
bool verify_sorted_records(
    const Record *records_array,
    const Record *original_records_array_copy_for_checksum,
    const Arguments &args) {
  if (args.N_elements == 0)
    return true;

  // Check 1: Non-decreasing order of keys
  for (size_t i = 0; i < args.N_elements - 1; ++i) {
    // if (current_rec->key > next_rec->key) { // CHANGED
    if (records_array[i].key > records_array[i + 1].key) { // USE ARRAY INDEXING
      if (!args.perf_mode) {
        std::cerr << "Correctness Error: Array not sorted at index " << i
                  << " and " << i + 1 << "." << std::endl;
        std::cerr << "  records[" << i
                  << "].key = " << records_array[i].key // USE ARRAY INDEXING
                  << std::endl;
        std::cerr << "  records[" << i + 1 << "].key = "
                  << records_array[i + 1].key // USE ARRAY INDEXING
                  << std::endl;
      }
      return false;
    }
  }

  // Check 2: (Optional) Data integrity using checksums if original data is
  // provided
  if (original_records_array_copy_for_checksum != nullptr) {
    unsigned long original_key_sum = 0;
    unsigned long sorted_key_sum = 0;

    for (size_t i = 0; i < args.N_elements; ++i) {
      // original_key_sum += (reinterpret_cast<const Record
      // *>(orig_ptr_char))->key; // CHANGED
      original_key_sum +=
          original_records_array_copy_for_checksum[i].key; // USE ARRAY INDEXING
      // sorted_key_sum += (reinterpret_cast<const Record
      // *>(sorted_ptr_char))->key; // CHANGED
      sorted_key_sum += records_array[i].key; // USE ARRAY INDEXING
    }

    if (original_key_sum != sorted_key_sum) {
      if (!args.perf_mode) {
        std::cerr << "Correctness Error: Sum of keys mismatch. Original="
                  << original_key_sum << ", Sorted=" << sorted_key_sum
                  << ". Possible data loss or corruption." << std::endl;
      }
      return false;
    }
  }

  if (!args.perf_mode) {
    std::cout << "Correctness check: Array is sorted." << std::endl;
    if (original_records_array_copy_for_checksum != nullptr) {
      std::cout << "Correctness check: Key sum matches." << std::endl;
    }
  }
  return true;
}

// Prints a sample of records.
void print_records_sample(const Record *records_array, const Arguments &args,
                          size_t sample_size) {
  if (args.perf_mode)
    return;

  std::cout << "Printing sample of " << std::min(sample_size, args.N_elements)
            << " records:" << std::endl;

  for (size_t i = 0; i < std::min(sample_size, args.N_elements); ++i) {
    const Record *rec = &records_array[i]; // USE ARRAY INDEXING
    std::cout << "  Record[" << i << "]: Key=" << rec->key << ", Payload=[";
    // Print first few bytes of payload for brevity
    for (size_t j = 0; j < std::min(args.R_payload_size_bytes, (size_t)8);
         ++j) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << (static_cast<int>(rec->rpayload[j]) & 0xFF) << " ";
    }
    if (args.R_payload_size_bytes > 8) {
      std::cout << "...";
    }
    std::cout << std::dec << "]" << std::endl;
  }
}
