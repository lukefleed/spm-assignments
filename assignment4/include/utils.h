#ifndef UTILS_H
#define UTILS_H

#include "record.h"
#include <string>
#include <vector> // Used by parse_arguments if it were to handle multiple inputs, etc.

// Structure to hold parsed command-line arguments
struct Arguments {
  size_t N_elements = 0;
  size_t R_payload_size_bytes = 0;
  int T_ff_threads = 0; // For FastFlow threads
  // Add P_mpi_processes for hybrid if needed
  // int P_mpi_processes = 0;
  std::string input_file_path;
  std::string output_file_path;
  bool check_correctness = false;
  bool perf_mode = false;
  unsigned long random_seed = 0; // Changed to unsigned long for consistency
};

bool parse_arguments(int argc, char *argv[], Arguments &args);
void generate_random_records(Record *records_array, const Arguments &args);
bool load_records_from_file(Record *records_array, const Arguments &args);
bool save_records_to_file(const Record *records_array, const Arguments &args);
bool verify_sorted_records(
    const Record *records_array,
    const Record *original_records_array_copy_for_checksum,
    const Arguments &args);
void print_records_sample(const Record *records_array, const Arguments &args,
                          size_t sample_size);

#endif // UTILS_H
