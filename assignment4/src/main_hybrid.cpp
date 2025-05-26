#include <iostream>
#include <mpi.h> // Include MPI header for MPI_Init and MPI_Finalize

int main(int argc, char *argv[]) {
  // Initialize MPI environment.
  // This is essential for any MPI program.
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Actual argument parsing and program logic will be added later.
  // For now, only rank 0 prints.
  if (world_rank == 0) {
    std::cout << "Hybrid MPI+FastFlow MergeSort Executable (main_hybrid.cpp)"
              << std::endl;
  }

  // Placeholder for future logic

  // Finalize the MPI environment.
  // This is essential and should be the last MPI call.
  MPI_Finalize();
  return 0;
}
