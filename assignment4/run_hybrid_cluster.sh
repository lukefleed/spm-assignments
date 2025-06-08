#!/bin/bash

## @file run_hybrid_cluster.sh
## @brief SLURM script for hybrid MPI+FastFlow performance testing with dual baseline analysis
## @details Performance analysis with sequential and parallel baselines.
##          Cluster constraints: one run per job, max 5min wall-time.
##          Speedup metrics: vs sequential std::sort and vs single-node parallel.
## @usage ./run_hybrid_cluster.sh <nodes>
## @example ./run_hybrid_cluster.sh 1    # baseline establishment
## @example ./run_hybrid_cluster.sh 2    # MPI scaling test

# ============================================================================
#                          CONFIGURATION PARAMETERS
# ============================================================================
# Modify these values before running on cluster with vim

# Performance test parameters
FF_THREADS=8       # FastFlow threads per MPI process
RECORDS_SIZE_M=100    # Array size in millions of records (5M = ~320MB per process)
PAYLOAD_SIZE_B=8  # Payload size in bytes

# Output file
CSV_FILENAME="hybrid_performance_results.csv"

# ============================================================================

# Check if node count parameter is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <nodes>"
    echo ""
    echo "Examples:"
    echo "  $0 1    # Dual baseline measurement (sequential + single-node parallel)"
    echo "  $0 2    # MPI scaling test (2 processes)"
    echo "  $0 4    # MPI scaling test (4 processes)"
    echo "  $0 8    # MPI scaling test (8 processes)"
    echo ""
    echo "Analysis: Sequential speedup, parallel speedup, MPI efficiency, total efficiency"
    echo ""
    echo "Professor's guidelines: Run ONE test per job, queue jobs one at a time"
    exit 1
fi

NODES=$1

# Validate input
if ! [[ "$NODES" =~ ^[0-9]+$ ]] || [ "$NODES" -lt 1 ]; then
    echo "Error: Number of nodes must be a positive integer"
    exit 1
fi

echo "======================================================================================================"
echo "               Hybrid MPI+FastFlow Performance Test Configuration                                   "
echo "======================================================================================================"
echo "Nodes:          ${NODES}"
echo "FF Threads:     ${FF_THREADS}"
echo "Array Size:     ${RECORDS_SIZE_M}M records"
echo "Payload Size:   ${PAYLOAD_SIZE_B}B"
echo "CSV Output:     ${CSV_FILENAME}"
echo "======================================================================================================"
echo "Started at: $(date)"

# Configure FastFlow on the required nodes before running the actual test
echo "Configuring FastFlow topology on ${NODES} node(s)..."
srun --nodes=${NODES} \
     --ntasks-per-node=1 \
     --time=00:02:00 \
     --mpi=pmix \
     bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh" >/dev/null 2>&1

echo "FastFlow configuration completed. Starting performance test..."

# Check if CSV file exists to determine if header is needed
FILE_EXISTS=false
if [ -f "${CSV_FILENAME}" ]; then
    FILE_EXISTS=true
fi

# Print comprehensive test header with dual baseline analysis
if [ "$FILE_EXISTS" = false ]; then
    echo "======================================================================================================"
    echo "                    Hybrid MPI+FastFlow Performance Analysis                                        "
    echo "------------------------------------------------------------------------------------------------------"
    echo " Config: ${RECORDS_SIZE_M}M records, ${PAYLOAD_SIZE_B}B payload, ${FF_THREADS} FF threads/process"
    echo " Analysis: Dual baseline comparison with MPI scaling"
    echo "======================================================================================================"
    echo "MPI Procs   Time (ms)      Throughput (MRec/s)   Seq Speedup  Par Speedup  MPI Eff (%)  Total Eff (%)"
    echo "----------- -------------- ------------------- ------------ ------------ ------------- --------------"
fi

# Run single test based on nodes parameter with enhanced baseline analysis
if [ ${NODES} -eq 1 ]; then
    # Baseline measurements with comprehensive analysis
    echo "=== Running dual baseline analysis: Sequential + Single-node Parallel ==="
    srun --ntasks=1 \
         --nodes=1 \
         --cpus-per-task=${FF_THREADS} \
         --time=00:03:00 \
         --mpi=pmix \
         bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet
else
    # Scaling test with dual speedup analysis
    echo "=== Running MPI scaling test: ${NODES} processes with dual baseline comparison ==="
    srun --nodes=${NODES} \
         --ntasks=${NODES} \
         --ntasks-per-node=1 \
         --time=00:05:00 \
         --mpi=pmix \
         bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet --skip-baselines
fi

echo ""
echo "------------------------------------------------------------------------------------------------------"
echo " Analysis completed for ${NODES} nodes. Results with dual speedup metrics saved to: ${CSV_FILENAME}"
echo " Metrics explanation:"
echo "   • Seq Speedup: Performance vs pure sequential std::sort baseline"
echo "   • Par Speedup: Performance vs single-node parallel (1 MPI + ${FF_THREADS} FF threads)"
echo "   • MPI Eff (%): MPI scaling efficiency (Par Speedup / MPI Processes)"
echo "   • Total Eff (%): Overall parallel efficiency (Seq Speedup / Total Threads)"
echo ""
echo " To run complete scaling analysis:"
echo "   ./run_hybrid_cluster.sh 1    # establish dual baselines"
echo "   ./run_hybrid_cluster.sh 2    # 2-node MPI scaling"
echo "   ./run_hybrid_cluster.sh 4    # 4-node MPI scaling"
echo "   ./run_hybrid_cluster.sh 8    # 8-node MPI scaling"

echo "Completed at: $(date)"
echo "======================================================================================================"
