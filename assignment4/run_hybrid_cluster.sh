#!/bin/bash

## @file run_hybrid_cluster.sh
## @brief SLURM script for hybrid MPI+FastFlow performance testing on cluster
## @details Follows strict cluster guidelines: one run per job, max 5min wall-time
## @usage ./run_hybrid_cluster.sh <nodes>
## @example ./run_hybrid_cluster.sh 1    # baseline
## @example ./run_hybrid_cluster.sh 2    # test with 2 nodes

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
    echo "  $0 1    # Baseline measurement (1 MPI process on 1 node)"
    echo "  $0 2    # Scaling test (2 MPI processes on 2 nodes)"
    echo "  $0 4    # Scaling test (4 MPI processes on 4 nodes)"
    echo "  $0 8    # Scaling test (8 MPI processes on 8 nodes)"
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

echo "==================================================================="
echo "         Hybrid MPI+FastFlow Performance Test Configuration        "
echo "==================================================================="
echo "Nodes:          ${NODES}"
echo "FF Threads:     ${FF_THREADS}"
echo "Array Size:     ${RECORDS_SIZE_M}M records"
echo "Payload Size:   ${PAYLOAD_SIZE_B}B"
echo "CSV Output:     ${CSV_FILENAME}"
echo "==================================================================="
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

# Print test header only if starting fresh or if this is first run
if [ "$FILE_EXISTS" = false ]; then
    echo "=============================================================================="
    echo "           Scientific MPI Scaling Analysis - Hybrid MergeSort                 "
    echo "------------------------------------------------------------------------------"
    echo " Config: ${RECORDS_SIZE_M}M records, ${PAYLOAD_SIZE_B}B payload, ${FF_THREADS} parallel threads/process"
    echo "=============================================================================="
    echo "MPI Procs   Time (ms)      Throughput (MRec/s)   Speedup      Efficiency (%)"
    echo "----------- -------------- ------------------- ------------ ---------------"
fi

# Run single test based on nodes parameter
if [ ${NODES} -eq 1 ]; then
    # Baseline measurement with 1 MPI process
    echo "=== Running baseline: 1 MPI process on 1 node ==="
    srun --ntasks=1 \
         --nodes=1 \
         --cpus-per-task=${FF_THREADS} \
         --time=00:03:00 \
         --mpi=pmix \
         bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet
else
    # Scaling test with specified number of nodes
    echo "=== Running scaling test: ${NODES} MPI processes on ${NODES} nodes ==="
    srun --nodes=${NODES} \
         --ntasks=${NODES} \
         --ntasks-per-node=1 \
         --time=00:05:00 \
         --mpi=pmix \
         bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet
fi

echo ""
echo "------------------------------------------------------------------------------"
echo " Test completed for ${NODES} nodes. Results appended to: ${CSV_FILENAME}"
echo " To run full scaling analysis, execute:"
echo "   ./run_hybrid_cluster.sh 1    # baseline"
echo "   ./run_hybrid_cluster.sh 2    # 2 nodes"
echo "   ./run_hybrid_cluster.sh 4    # 4 nodes"
echo "   ./run_hybrid_cluster.sh 8    # 8 nodes"

echo "Completed at: $(date)"
echo "==================================================================="
