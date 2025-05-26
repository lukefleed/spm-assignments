#ifndef UTILS_H
#define UTILS_H

#include "record.h" // For struct Record
#include <cstddef>  // For size_t
#include <string>
#include <vector>

// Structure to hold parsed command-line arguments for both executables.
// This can be expanded as more options are needed.
struct Arguments {
  size_t N_elements = 0;           // Number of records to sort.
  size_t R_payload_size_bytes = 0; // Payload size for each record in bytes.
  int T_ff_threads = 0;            // Number of FastFlow threads.
  // int P_mpi_processes = 0; // This will be handled by MPI runtime, but useful
  // for root node info.
  std::string input_file_path;    // Path to an input data file.
  std::string output_file_path;   // Path for an output data file (optional).
  bool check_correctness = false; // Flag to enable correctness verification.
  bool perf_mode = false; // Flag for minimal output, suitable for scripting.
  unsigned int random_seed = 0; // Seed for random data generation.
};

// Parses command line arguments for the application.
// Populates the Arguments struct.
// Returns true if parsing is successful, false on error or if help is
// requested.
bool parse_arguments(int argc, char *argv[], Arguments &args);

// Generates an array of records with random keys and payloads.
// The actual payload content per record will be 'args.R_payload_size_bytes'.
// The memory for 'records_array' must be pre-allocated by the caller to hold
// 'args.N_elements'. Each record in the array will have
// 'get_record_actual_size(args.R_payload_size_bytes)' bytes.
void generate_random_records(Record *records_array, const Arguments &args);

// Loads records from a binary file.
// The file is expected to contain N_elements records, each with a key
// and R_payload_size_bytes of payload.
// Returns true on success, false on failure (e.g., file not found, read error).
// The memory for 'records_array' must be pre-allocated.
bool load_records_from_file(Record *records_array, const Arguments &args);

// Saves records to a binary file.
// Writes N_elements records to the specified file.
// Returns true on success, false on failure (e.g., cannot open file, write
// error).
bool save_records_to_file(const Record *records_array, const Arguments &args);

// Verifies if the given array of records is sorted correctly based on keys.
// Also optionally checks if elements were lost or duplicated (basic check).
// Returns true if sorted, false otherwise.
bool verify_sorted_records(
    const Record *records_array,
    const Record
        *original_records_array_copy_for_checksum, // Optional, for integrity
    const Arguments &args);

// Prints a small sample of records for debugging or verification.
void print_records_sample(const Record *records_array, const Arguments &args,
                          size_t sample_size = 10);

#endif // UTILS_H
