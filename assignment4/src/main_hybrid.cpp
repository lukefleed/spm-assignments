#include "mergesort_common.h"  // For copy_record (used by utils and merge)
#include "mergesort_ff.h"      // For parallel_merge_sort_ff (local sort)
#include "mergesort_hybrid.h"  // For k-way merge (in later phases)
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

  // --- Argument Parsing (Rank 0) and Broadcasting ---
  // [ कोड Fase 1 e 2 qui - Identico a prima ]
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
  if (args.N_elements == 0 && world_rank == 0) {
  } else if (args.N_elements == 0 && world_rank != 0) {
    MPI_Finalize();
    return 0;
  } else if (args.N_elements < 0 && world_rank == 0) {
    std::cerr << "N_elements <0 impossible" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
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
  MPI_Datatype MPI_Record_ActivePart_Type;
  MPI_Datatype MPI_Record_Resized_Type;
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
  MPI_Type_create_struct(num_record_members, blocklengths, displacements, types,
                         &MPI_Record_ActivePart_Type);
  MPI_Type_create_resized(MPI_Record_ActivePart_Type, 0, sizeof(Record),
                          &MPI_Record_Resized_Type);
  MPI_Type_commit(&MPI_Record_Resized_Type);
  MPI_Type_free(&MPI_Record_ActivePart_Type);
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
    std::cout << "  Expected active part size:   " << expected_active_part_size
              << " bytes." << std::endl;
    if (active_extent != sizeof(Record)) {
      std::cout
          << "  WARNING: Resized MPI type extent does not match sizeof(Record)."
          << std::endl;
    }
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of MPI Datatype Definition ---

  PerformanceTimer total_timer;
  if (world_rank == 0) {
    total_timer.start();
  }

  // --- Data Loading/Generation (Rank 0) & Memory Allocation ---
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

  // --- Calculate Per-Process Element Counts and Displacements for Scatterv ---
  std::vector<int> sendcounts_scatter(world_size);
  std::vector<int> displs_scatter(world_size);
  if (args.N_elements > 0) {
    size_t base_chunk_size = args.N_elements / world_size;
    size_t remainder_elements = args.N_elements % world_size;
    int current_displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts_scatter[i] =
          static_cast<int>(base_chunk_size + (i < remainder_elements ? 1 : 0));
      displs_scatter[i] = current_displacement;
      current_displacement += sendcounts_scatter[i];
    }
  } else {
    for (int i = 0; i < world_size; ++i) {
      sendcounts_scatter[i] = 0;
      displs_scatter[i] = 0;
    }
  }
  size_t local_n_elements = static_cast<size_t>(sendcounts_scatter[world_rank]);

  // --- DEBUG: Print global data sample on Rank 0 BEFORE Scatterv ---
  if (world_rank == 0 && !args.perf_mode && args.N_elements > 0 &&
      records_array_global_uptr) {
    std::cout << "DEBUG Rank 0: Sample of global data BEFORE Scatterv:"
              << std::endl;
    size_t print_limit = std::min((size_t)15, args.N_elements);
    for (size_t i = 0; i < print_limit; ++i) {
      std::cout << "  Global[" << i
                << "].key = " << records_array_global_uptr[i].key;
      for (int r = 0; r < world_size; ++r) {
        if (static_cast<int>(i) >= displs_scatter[r] &&
            static_cast<int>(i) < (displs_scatter[r] + sendcounts_scatter[r])) {
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

  // --- Scatter Data from Rank 0 to All Processes ---
  if (args.N_elements > 0) {
    MPI_Scatterv(
        (world_rank == 0 ? records_array_global_uptr.get() : nullptr),
        sendcounts_scatter.data(), displs_scatter.data(),
        MPI_Record_Resized_Type,
        (local_n_elements > 0 ? local_records_array_uptr.get() : nullptr),
        static_cast<int>(local_n_elements), MPI_Record_Resized_Type, 0,
        MPI_COMM_WORLD);
  }

  // --- Print received data sample (Debug) ---
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
  if (local_n_elements > 0 && local_records_array_uptr) {
    if (!args.perf_mode) {
      // Optional: Print first element before sort for this rank
      // std::cout << "Rank " << world_rank << " BEFORE local sort, first key: "
      // << local_records_array_uptr[0].key << std::endl;
    }

    parallel_merge_sort_ff(local_records_array_uptr.get(), local_n_elements,
                           args.R_payload_size_bytes, args.T_ff_threads);

    if (!args.perf_mode) {
      MPI_Barrier(MPI_COMM_WORLD); // Wait for all sorts to finish before
                                   // printing status
      // Optional: Print first element after sort for this rank
      // std::cout << "Rank " << world_rank << " AFTER local sort, first key: "
      // << local_records_array_uptr[0].key << std::endl;
    }
  }
  // Synchronize all processes after local sort
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0 && !args.perf_mode && args.N_elements > 0) {
    std::cout << "Local parallel_merge_sort_ff completed on all ranks."
              << std::endl;
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }
  // --- End of Local Sort ---

  // --- Placeholder for Gatherv (Fase 5) ---
  // std::unique_ptr<Record[]> gathered_chunks_buffer_uptr = nullptr;
  // if (world_rank == 0 && args.N_elements > 0) {
  //    gathered_chunks_buffer_uptr =
  //    allocate_record_array_main(args.N_elements, "gathered_chunks_buffer",
  //    world_rank);
  // }
  // if (args.N_elements > 0) {
  //    MPI_Gatherv( ... );
  // }
  // --- End of Placeholder for Gatherv ---

  // --- Placeholder for Final Merge (Fase 6 & 7) ---
  // if (world_rank == 0 && args.N_elements > 0) {
  //    total_timer.stop();
  //    sequential_k_way_merge_on_root(gathered_chunks_buffer_uptr.get(), ...,
  //    sorted_records_global_uptr.get(), ...);
  //    // Print time, verify, save
  // } else if (world_rank == 0 && args.N_elements == 0) {
  //    total_timer.stop(); // Stop timer even if N=0
  //    if(!args.perf_mode) std::cout << "N_elements is 0. No sorting performed.
  //    Total time: " << total_timer.elapsed_seconds() << "s" << std::endl; else
  //    std::cout << args.N_elements << "," << args.R_payload_size_bytes << ","
  //    << args.T_ff_threads << "," << world_size << "," <<
  //    total_timer.elapsed_seconds() << std::endl; // CSV for N=0
  // }
  // --- End of Placeholder for Final Merge ---

  // Cleanup and Finalize
  MPI_Type_free(&MPI_Record_Resized_Type);
  MPI_Finalize();
  return 0;
}
