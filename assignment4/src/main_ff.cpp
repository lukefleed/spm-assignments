#include "mergesort_common.h" // For sequential_merge_sort_recursive, Record, get_record_actual_size, merge_records
#include "mergesort_ff.h"
#include "performance_timer.h"
#include "record.h"
#include "utils.h"

#include <ff/ff.hpp>

#include <iostream>
#include <memory> // For std::unique_ptr for robust memory management
#include <string>
#include <vector>

// Helper function to manage memory allocation and deallocation for Record
// arrays This avoids raw new/delete in main and handles potential bad_alloc.
std::unique_ptr<Record[]> allocate_record_array(size_t num_elements,
                                                size_t r_payload_size_bytes) {
  // Calculate the size of one record instance based on actual payload.
  // While Record struct has MAX_RPAYLOAD_SIZE, we're interested in the
  // memory footprint based on the runtime R_payload_size_bytes for the array.
  // However, C++ allocates structs based on their declared size (Record).
  // Pointer arithmetic within sort functions will use get_record_actual_size.
  // For the array allocation itself, `new Record[num_elements]` is sufficient.
  // The custom copy_record and memory operations in sorts handle the actual
  // payload size.
  try {
    return std::make_unique<Record[]>(num_elements);
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed for " << num_elements
              << " records: " << e.what() << std::endl;
    return nullptr;
  }
}

int main(int argc, char *argv[]) {
  Arguments args;
  if (!parse_arguments(argc, argv, args)) {
    // parse_arguments already prints usage/error messages.
    return 1;
  }

  if (args.T_ff_threads <= 0) {
    int detected_cores = ff_numCores();
    if (detected_cores <= 0)
      detected_cores = 1; // Fallback if detection fails

    if (!args.perf_mode) {
      std::cout << "Number of FastFlow threads (-t) not specified or invalid. "
                   "Defaulting to system's available cores: "
                << detected_cores << std::endl;
    }
    args.T_ff_threads = detected_cores;
  }

  if (!args.perf_mode) {
    std::cout << "Configuration: N=" << args.N_elements
              << ", R=" << args.R_payload_size_bytes
              << ", T_FF=" << args.T_ff_threads << std::endl;
    if (!args.input_file_path.empty()) {
      std::cout << "Input file: " << args.input_file_path << std::endl;
    } else {
      std::cout << "Generating random data with seed: " << args.random_seed
                << std::endl;
    }
  }

  std::unique_ptr<Record[]> records_uptr =
      allocate_record_array(args.N_elements, args.R_payload_size_bytes);
  if (!records_uptr) {
    return 1; // Allocation failure already reported
  }
  Record *records_array = records_uptr.get();

  if (!args.input_file_path.empty()) {
    if (!load_records_from_file(records_array, args)) {
      std::cerr << "Failed to load records from file." << std::endl;
      return 1;
    }
  } else {
    generate_random_records(records_array, args);
  }

  std::unique_ptr<Record[]> original_records_copy_uptr = nullptr;
  if (args.check_correctness && args.N_elements > 0) {
    original_records_copy_uptr =
        allocate_record_array(args.N_elements, args.R_payload_size_bytes);
    if (!original_records_copy_uptr) {
      std::cerr << "Failed to allocate memory for correctness check copy."
                << std::endl;
      return 1;
    }
    // Deep copy the records for verification using copy_record
    Record *src_array_ptr = records_array;
    Record *dst_array_ptr = original_records_copy_uptr.get();
    for (size_t i = 0; i < args.N_elements; ++i) {
      copy_record(&dst_array_ptr[i], &src_array_ptr[i],
                  args.R_payload_size_bytes);
    }
  }

  PerformanceTimer timer;
  timer.start();

  parallel_merge_sort_ff(records_array, args.N_elements,
                         args.R_payload_size_bytes, args.T_ff_threads);

  timer.stop();

  if (args.perf_mode) {
    std::cout << args.N_elements << "," << args.R_payload_size_bytes << ","
              << args.T_ff_threads << "," << timer.elapsed_seconds()
              << std::endl;
  } else {
    std::cout << "Sorting completed." << std::endl;
    std::cout << "Elapsed time: " << timer.elapsed_seconds() << " seconds ("
              << timer.elapsed_milliseconds() << " ms)." << std::endl;

    if (args.N_elements > 0 &&
        args.N_elements <= 20) { // Print all if very small, otherwise sample
      print_records_sample(records_array, args, args.N_elements);
    } else if (args.N_elements > 0) {
      print_records_sample(records_array, args, 10); // Print sample of 10
    }
  }

  if (args.check_correctness) {
    if (!args.perf_mode) { // Provide context for the check
      std::cout << "Verifying sort correctness..." << std::endl;
    }
    bool sorted_correctly = verify_sorted_records(
        records_array, original_records_copy_uptr.get(), args);
    if (!args.perf_mode) {
      if (sorted_correctly) {
        std::cout << "Correctness check: PASSED." << std::endl;
      } else {
        std::cout << "Correctness check: FAILED." << std::endl;
      }
    }
    if (!sorted_correctly &&
        args.perf_mode) { // Still output error code for scripts
      return 1;           // Indicate failure
    }
  }

  if (!args.output_file_path.empty()) {
    if (!args.perf_mode) {
      std::cout << "Saving sorted records to: " << args.output_file_path
                << std::endl;
    }
    if (!save_records_to_file(records_array, args)) {
      std::cerr << "Failed to save records to file." << std::endl;
      // Not returning error here, as sorting might have been successful
    }
  }

  // Memory for records_array and original_records_copy_uptr is managed by
  // unique_ptr
  return 0;
}
