#include "mergesort_common.h"  // For copy_record (used by utils and merge)
#include "mergesort_ff.h"      // For parallel_merge_sort_ff (local sort)
#include "mergesort_hybrid.h"  // For sequential_k_way_merge_on_root
#include "performance_timer.h" // For PerformanceTimer
#include "record.h"
#include "utils.h" // For parse_arguments, generate/load/save, verify, print_sample

#include <algorithm> // For std::min
#include <cstring>
#include <iostream>
#include <memory> // For std::unique_ptr
#include <mpi.h>
#include <numeric> // For std::accumulate
#include <string>
#include <vector>

// Helper function to manage memory allocation for Record arrays
std::unique_ptr<Record[]>
allocate_record_array_main(size_t num_elements, const char *array_name_debug,
                           int current_rank, bool abort_on_fail = true) {
  try {
    if (num_elements == 0) {
      if (std::string(array_name_debug).find("global") != std::string::npos &&
          current_rank == 0) {
      }
      return nullptr;
    }
    return std::make_unique<Record[]>(num_elements);
  } catch (const std::bad_alloc &e) {
    if (current_rank == 0 ||
        std::string(array_name_debug).find("local") != std::string::npos ||
        std::string(array_name_debug).find("gathered") != std::string::npos) {
      std::cerr << "Rank " << current_rank << ": Memory allocation failed for "
                << array_name_debug << " (" << num_elements
                << " records): " << e.what() << std::endl;
    }
    if (abort_on_fail) {
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return nullptr;
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Arguments args;
  std::unique_ptr<Record[]> records_array_global_uptr = nullptr;
  std::unique_ptr<Record[]> sorted_records_global_uptr = nullptr;
  std::unique_ptr<Record[]> original_records_copy_uptr = nullptr;
  std::unique_ptr<Record[]> gathered_chunks_buffer_uptr = nullptr;

  // --- Argument Parsing (Rank 0) and Broadcasting ---
  // [ Identical to previous version ]
  if (world_rank == 0) {
    if (!parse_arguments(argc, argv, args)) {
      std::cerr << "Rank 0: Argument parsing failed. Aborting." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }
  MPI_Bcast(&args.N_elements, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.R_payload_size_bytes, 1, MPI_UNSIGNED_LONG, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(&args.T_ff_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.random_seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.check_correctness, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&args.perf_mode, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  char input_file_path_char[256] = {0};
  char output_file_path_char[256] = {0};
  if (world_rank == 0) {
    if (!args.input_file_path.empty()) {
      strncpy(input_file_path_char, args.input_file_path.c_str(),
              sizeof(input_file_path_char) - 1);
      input_file_path_char[sizeof(input_file_path_char) - 1] = '\0';
    }
    if (!args.output_file_path.empty()) {
      strncpy(output_file_path_char, args.output_file_path.c_str(),
              sizeof(output_file_path_char) - 1);
      output_file_path_char[sizeof(output_file_path_char) - 1] = '\0';
    }
  }
  MPI_Bcast(input_file_path_char, sizeof(input_file_path_char), MPI_CHAR, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(output_file_path_char, sizeof(output_file_path_char), MPI_CHAR, 0,
            MPI_COMM_WORLD);
  if (world_rank != 0) {
    if (input_file_path_char[0] != '\0') {
      args.input_file_path = std::string(input_file_path_char);
    } else {
      args.input_file_path.clear();
    }
    if (output_file_path_char[0] != '\0') {
      args.output_file_path = std::string(output_file_path_char);
    } else {
      args.output_file_path.clear();
    }
  }
  if (args.N_elements == 0 &&
      world_rank == 0) { /* N=0 is handled as a valid case later */
  } else if (args.N_elements == 0 && world_rank != 0) {
    MPI_Type_free(
        nullptr); /* Type might not be created if N=0 path is taken early */
    MPI_Finalize();
    return 0;
  }
  if (args.R_payload_size_bytes > MAX_RPAYLOAD_SIZE) {
    if (world_rank == 0) {
      std::cerr << "Error: R_payload_size (" << args.R_payload_size_bytes
                << ") > MAX_RPAYLOAD_SIZE (" << MAX_RPAYLOAD_SIZE
                << "). Aborting." << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  // --- End of Argument Parsing and Broadcasting ---

  // --- Print Configuration (Rank 0) ---
  // [ Identical to previous version ]
  if (world_rank == 0 && !args.perf_mode) {
    std::cout << "-----------------------------------------------------"
              << std::endl;
    std::cout << "MPI Hybrid MergeSort Configuration:" << std::endl;
    std::cout << "  Total MPI Processes (P_MPI): " << world_size << std::endl;
    std::cout << "  Array Size (N_elements):     " << args.N_elements
              << std::endl;
    std::cout << "  Record Payload (R_bytes):    " << args.R_payload_size_bytes
              << std::endl;
    std::cout << "  FastFlow Threads (T_FF/node):" << args.T_ff_threads
              << std::endl;
    if (!args.input_file_path.empty()) {
      std::cout << "  Input File:                  " << args.input_file_path
                << std::endl;
    } else {
      std::cout << "  Input Data:                  Generated Randomly"
                << std::endl;
      std::cout << "  Random Seed:                 " << args.random_seed
                << std::endl;
    }
    if (!args.output_file_path.empty()) {
      std::cout << "  Output File:                 " << args.output_file_path
                << std::endl;
    }
    std::cout << "  Correctness Check:           "
              << (args.check_correctness ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Performance Mode:            "
              << (args.perf_mode ? "Enabled" : "Disabled") << std::endl;
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of Print Configuration ---

  // --- MPI Datatype Definition for Record ---
  // [ Identical to previous version ]
  MPI_Datatype MPI_Record_ActivePart_Type;
  MPI_Datatype MPI_Record_Resized_Type =
      MPI_DATATYPE_NULL;     // Initialize to NULL for conditional free
  if (args.N_elements > 0) { // Only define if there is data to transfer
    const int num_record_members = 2;
    int blocklengths[num_record_members];
    MPI_Aint displacements[num_record_members];
    MPI_Datatype types[num_record_members];
    blocklengths[0] = 1;
    types[0] = MPI_UNSIGNED_LONG;
    Record dummy_record_for_layout;
    MPI_Aint base_address;
    MPI_Get_address(&dummy_record_for_layout, &base_address);
    MPI_Get_address(&dummy_record_for_layout.key, &displacements[0]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    blocklengths[1] = static_cast<int>(args.R_payload_size_bytes);
    types[1] = MPI_CHAR;
    MPI_Get_address(&dummy_record_for_layout.rpayload[0], &displacements[1]);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    MPI_Type_create_struct(num_record_members, blocklengths, displacements,
                           types, &MPI_Record_ActivePart_Type);
    MPI_Type_create_resized(MPI_Record_ActivePart_Type, 0, sizeof(Record),
                            &MPI_Record_Resized_Type);
    MPI_Type_commit(&MPI_Record_Resized_Type);
    MPI_Type_free(&MPI_Record_ActivePart_Type); // Free the intermediate type
    if (world_rank == 0 && !args.perf_mode) {
      MPI_Aint active_extent, active_lb;
      MPI_Type_get_extent(MPI_Record_Resized_Type, &active_lb, &active_extent);
      size_t expected_active_part_size =
          sizeof(unsigned long) + args.R_payload_size_bytes;
      std::cout << "MPI_Record_Resized_Type defined. Describes active members "
                   "but with full C++ struct extent."
                << std::endl;
      std::cout << "  MPI Type (Resized) Extent:   " << active_extent
                << " bytes. (Should match sizeof(Record))" << std::endl;
      std::cout << "  sizeof(Record) in C++:       " << sizeof(Record)
                << " bytes." << std::endl;
      std::cout << "  Expected active part size:   "
                << expected_active_part_size << " bytes." << std::endl;
      if (active_extent != sizeof(Record)) {
        std::cout << "  WARNING: Resized MPI type extent does not match "
                     "sizeof(Record)."
                  << std::endl;
      }
      std::cout << "-----------------------------------------------------"
                << std::endl;
    }
  }
  // --- End of MPI Datatype Definition ---

  PerformanceTimer total_timer;
  if (world_rank == 0) {
    total_timer.start();
  }

  // --- Data Loading/Generation (Rank 0) & Memory Allocation ---
  // [ Identical to previous version ]
  if (world_rank == 0) {
    if (args.N_elements > 0) {
      records_array_global_uptr = allocate_record_array_main(
          args.N_elements, "global records_array", world_rank);
      if (!records_array_global_uptr) {
      }
      if (!args.input_file_path.empty()) {
        if (!load_records_from_file(records_array_global_uptr.get(), args)) {
          std::cerr << "Rank 0: Failed to load records. Aborting." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
          return 1;
        }
      } else {
        generate_random_records(records_array_global_uptr.get(), args);
      }
      if (args.check_correctness) {
        original_records_copy_uptr = allocate_record_array_main(
            args.N_elements, "original_records_copy", world_rank);
        if (!original_records_copy_uptr) {
        }
        Record *src_ptr = records_array_global_uptr.get();
        Record *dst_ptr = original_records_copy_uptr.get();
        for (size_t i = 0; i < args.N_elements; ++i) {
          copy_record(&dst_ptr[i], &src_ptr[i], args.R_payload_size_bytes);
        }
      }
      sorted_records_global_uptr = allocate_record_array_main(
          args.N_elements, "final sorted_records_global", world_rank);
    }
  }
  // --- End of Data Loading ---

  // --- Calculate Per-Process Element Counts and Displacements for
  // Scatterv/Gatherv --- [ Identical to previous version ]
  std::vector<int> counts_per_process(world_size);
  std::vector<int> displs_in_global_buffer(world_size);
  if (args.N_elements > 0) {
    size_t base_chunk_size = args.N_elements / world_size;
    size_t remainder_elements = args.N_elements % world_size;
    int current_displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      counts_per_process[i] =
          static_cast<int>(base_chunk_size + (i < remainder_elements ? 1 : 0));
      displs_in_global_buffer[i] = current_displacement;
      current_displacement += counts_per_process[i];
    }
  } else {
    for (int i = 0; i < world_size; ++i) {
      counts_per_process[i] = 0;
      displs_in_global_buffer[i] = 0;
    }
  }
  size_t local_n_elements = static_cast<size_t>(counts_per_process[world_rank]);
  // --- End of Count/Displacement Calculation ---

  // --- DEBUG: Print global data sample on Rank 0 BEFORE Scatterv ---
  // [ Identical to previous version ]
  if (world_rank == 0 && !args.perf_mode && args.N_elements > 0 &&
      records_array_global_uptr) {
    std::cout << "DEBUG Rank 0: Sample of global data BEFORE Scatterv:"
              << std::endl;
    size_t print_limit = std::min((size_t)15, args.N_elements);
    for (size_t i = 0; i < print_limit; ++i) {
      std::cout << "  Global[" << i
                << "].key = " << records_array_global_uptr[i].key;
      for (int r = 0; r < world_size; ++r) {
        if (static_cast<int>(i) >= displs_in_global_buffer[r] &&
            static_cast<int>(i) <
                (displs_in_global_buffer[r] + counts_per_process[r])) {
          std::cout << " (expected for rank " << r << ")";
          break;
        }
      }
      std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of DEBUG Print ---

  std::unique_ptr<Record[]> local_records_array_uptr =
      allocate_record_array_main(local_n_elements, "local_records_array",
                                 world_rank, true);

  // --- Scatter Data from Rank 0 to All Processes (Fase 3) ---
  // [ Identical to previous version ]
  if (args.N_elements > 0) {
    MPI_Scatterv(
        (world_rank == 0 ? records_array_global_uptr.get() : nullptr),
        counts_per_process.data(), displs_in_global_buffer.data(),
        MPI_Record_Resized_Type,
        (local_n_elements > 0 ? local_records_array_uptr.get() : nullptr),
        static_cast<int>(local_n_elements), MPI_Record_Resized_Type, 0,
        MPI_COMM_WORLD);
  }
  // --- End of Scatter ---

  // --- Print received data sample (Debug) ---
  // [ Identical to previous version ]
  if (!args.perf_mode && args.N_elements > 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < world_size; ++r) {
      if (world_rank == r) {
        std::cout << "Rank " << world_rank << " received " << local_n_elements
                  << " records after Scatterv.";
        if (local_n_elements > 0 && local_records_array_uptr) {
          std::cout << " Sample local data (first key): "
                    << local_records_array_uptr[0].key;
        }
        std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (world_rank == 0)
      std::cout << "-----------------------------------------------------"
                << std::endl;
  } else if (!args.perf_mode && args.N_elements == 0 && world_rank == 0) {
    std::cout
        << "N_elements is 0, Scatterv skipped. All local_n_elements are 0."
        << std::endl;
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of Print received data sample ---

  // --- Local Sort (Fase 4) ---
  // [ Identical to previous version ]
  if (local_n_elements > 0 && local_records_array_uptr) {
    parallel_merge_sort_ff(local_records_array_uptr.get(), local_n_elements,
                           args.R_payload_size_bytes, args.T_ff_threads);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0 && !args.perf_mode && args.N_elements > 0) {
    std::cout << "Local parallel_merge_sort_ff completed on all ranks."
              << std::endl;
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of Local Sort ---

  // --- Gather Sorted Data to Rank 0 (Fase 5) ---
  // [ Identical to previous version ]
  if (world_rank == 0 && args.N_elements > 0) {
    gathered_chunks_buffer_uptr = allocate_record_array_main(
        args.N_elements, "gathered_chunks_buffer", world_rank);
    if (!gathered_chunks_buffer_uptr) { /* Aborted */
    }
  }
  if (args.N_elements > 0) {
    MPI_Gatherv(
        (local_n_elements > 0 ? local_records_array_uptr.get() : nullptr),
        static_cast<int>(local_n_elements), MPI_Record_Resized_Type,
        (world_rank == 0 && gathered_chunks_buffer_uptr
             ? gathered_chunks_buffer_uptr.get()
             : nullptr),
        (world_rank == 0 ? counts_per_process.data() : nullptr),
        (world_rank == 0 ? displs_in_global_buffer.data() : nullptr),
        MPI_Record_Resized_Type, 0, MPI_COMM_WORLD);
  }
  if (world_rank == 0 && !args.perf_mode && args.N_elements > 0) {
    std::cout << "MPI_Gatherv completed. Rank 0 has concatenated sorted chunks."
              << std::endl;
    if (gathered_chunks_buffer_uptr) {
      std::cout << "DEBUG Rank 0: Sample of gathered data (first key of each "
                   "chunk expected):"
                << std::endl;
      for (int i = 0; i < world_size; ++i) {
        if (counts_per_process[i] > 0) {
          int start_offset_of_chunk = displs_in_global_buffer[i];
          std::cout << "  Chunk from original rank " << i
                    << " (starts at gathered_buffer[" << start_offset_of_chunk
                    << "]): First key = "
                    << gathered_chunks_buffer_uptr[start_offset_of_chunk].key
                    << std::endl;
        } else {
          std::cout << "  Chunk from original rank " << i << ": (empty)"
                    << std::endl;
        }
      }
    }
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of Gather Data ---

  // --- Final Merge, Timing, Verification, and Output (Fase 7) ---
  if (world_rank == 0) {
    if (args.N_elements > 0) {
      if (!gathered_chunks_buffer_uptr) {
        std::cerr << "Rank 0 Error: gathered_chunks_buffer is null before "
                     "final merge. Aborting."
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
      }
      if (!sorted_records_global_uptr) {
        std::cerr << "Rank 0 Error: sorted_records_global_uptr is null before "
                     "final merge. Aborting."
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
      }
      // Perform the k-way merge on rank 0
      sequential_k_way_merge_on_root(
          gathered_chunks_buffer_uptr.get(), args.N_elements,
          world_size, // Number of chunks is the number of MPI processes
          counts_per_process.data(),
          displs_in_global_buffer
              .data(), // These define the layout in gathered_chunks_buffer
          sorted_records_global_uptr.get(), args.R_payload_size_bytes);
      if (!args.perf_mode) {
        std::cout << "Sequential k-way merge completed on Rank 0." << std::endl;
      }
    } // End if (args.N_elements > 0) for merge

    total_timer.stop(); // Stop timer after all sorting work is done on rank 0

    if (args.perf_mode) {
      // CSV output: N,R,T_FF,P_MPI,time
      // This is the benchmark output format expected by run_benchmarks.sh for
      // hybrid mode
      std::cout << args.N_elements << "," << args.R_payload_size_bytes << ","
                << args.T_ff_threads << "," << world_size << ","
                << total_timer.elapsed_seconds() << std::endl;
    } else {
      std::cout << "-----------------------------------------------------"
                << std::endl;
      std::cout << "Total sorting time: " << total_timer.elapsed_seconds()
                << " seconds (" << total_timer.elapsed_milliseconds() << " ms)."
                << std::endl;

      if (args.N_elements > 0 && sorted_records_global_uptr) {
        if (args.N_elements <= 20) { // Print all if very small
          print_records_sample(sorted_records_global_uptr.get(), args,
                               args.N_elements);
        } else {
          print_records_sample(sorted_records_global_uptr.get(), args,
                               10); // Print sample of 10
        }
      }
    }

    bool sorted_correctly_flag = true; // Assume correct for N=0
    if (args.check_correctness && args.N_elements > 0) {
      if (!args.perf_mode) {
        std::cout << "Verifying sort correctness..." << std::endl;
      }
      if (!sorted_records_global_uptr || !original_records_copy_uptr) {
        std::cerr << "Rank 0 Error: Buffers for correctness check are null."
                  << std::endl;
        sorted_correctly_flag = false;
      } else {
        sorted_correctly_flag =
            verify_sorted_records(sorted_records_global_uptr.get(),
                                  original_records_copy_uptr.get(), args);
      }

      if (!args.perf_mode) {
        if (sorted_correctly_flag) {
          std::cout << "Correctness check: PASSED." << std::endl;
        } else {
          std::cout << "Correctness check: FAILED." << std::endl;
        }
      }
      if (!sorted_correctly_flag && args.perf_mode) {
        // For perf_mode, if correctness fails, we might want to indicate error
        // differently For now, the script will check return code of the
        // executable. To make script fail, we need to exit with non-zero.
        // MPI_Abort already called if memory fails. If verify fails, need to
        // decide action. Let's ensure main returns non-zero if check fails.
      }
    }

    if (!args.output_file_path.empty() && args.N_elements > 0 &&
        sorted_records_global_uptr) {
      if (!args.perf_mode) {
        std::cout << "Saving sorted records to: " << args.output_file_path
                  << std::endl;
      }
      if (!save_records_to_file(sorted_records_global_uptr.get(), args)) {
        std::cerr << "Rank 0: Failed to save records to file." << std::endl;
        // Not necessarily a fatal error for the sort itself
      }
    }
    if (!args.perf_mode)
      std::cout << "-----------------------------------------------------"
                << std::endl;

    // If correctness check failed, and we are in perf_mode, we should ensure
    // the run_benchmarks.sh script notices. The script checks the exit code.
    // MPI_Abort handles critical errors. For a failed correctness check that
    // doesn't abort, we need main to return non-zero.
    if (args.check_correctness && !sorted_correctly_flag &&
        args.N_elements > 0) {
      // Ensure MPI_Finalize is called by all, then rank 0 can exit with error.
      // However, MPI standard says only one process should call MPI_Finalize if
      // others might have aborted. This path is only for rank 0. To signal
      // error to script, we can't use MPI_Abort here as it kills others before
      // they finalize. Simplest is just to let main return non-zero if this
      // path is hit.
    }

  } // End if (world_rank == 0) for final steps

  // All processes must call finalize.
  // If rank 0 found a correctness error, it should still allow others to
  // finalize. The return code of main from rank 0 will be the one mpirun
  // reports (usually).
  if (MPI_Record_Resized_Type !=
      MPI_DATATYPE_NULL) { // Conditionally free the type
    MPI_Type_free(&MPI_Record_Resized_Type);
  }
  MPI_Finalize();

  if (world_rank == 0 && args.check_correctness && args.N_elements > 0) {
    // If verify_sorted_records returned false, sorted_correctly_flag will be
    // false. We need main to return non-zero.
    bool final_check_passed =
        true; // Re-evaluate based on verify_sorted_records for return code
    if (args.check_correctness && args.N_elements > 0) {
      if (sorted_records_global_uptr &&
          original_records_copy_uptr) { // Re-check pointers for safety
        final_check_passed = verify_sorted_records(
            sorted_records_global_uptr.get(), original_records_copy_uptr.get(),
            args); // Re-run verify just to get the boolean if not stored, or
                   // use stored flag
      } else {
        final_check_passed = false; // Buffers were null, so check is false
      }
    }
    return final_check_passed
               ? 0
               : 1; // Return 0 on success, 1 on correctness failure
  }

  return 0; // Default success for non-rank0 or if no correctness check
}
