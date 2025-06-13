#!/bin/bash

# Default parameters (can be overridden via command line)
DEFAULT_FF_THREADS=10
DEFAULT_RECORDS_SIZE_M=100
DEFAULT_PAYLOAD_SIZE_B=16
DEFAULT_CSV_FILENAME="hybrid_performance_results.csv"

# ============================================================================

show_usage() {
    cat << EOF
Usage: $0 "node_list" [ff_threads] [records_m] [payload_b] [csv_filename]

Parameters:
  node_list     Space-separated list of node counts (quoted)
  ff_threads    FastFlow threads per MPI process (default: ${DEFAULT_FF_THREADS})
  records_m     Dataset size in millions of records (default: ${DEFAULT_RECORDS_SIZE_M})
  payload_b     Payload size in bytes (default: ${DEFAULT_PAYLOAD_SIZE_B})
  csv_filename  Output CSV file (default: ${DEFAULT_CSV_FILENAME})

Examples:
  $0 "1 2 4 8"                    # Full scaling test with defaults
  $0 "1 2 4" 16 50 64             # Custom parameters
  $0 "1 2" 8 10 8 "test.csv"      # All custom parameters
EOF
}

# Validate and parse arguments
if [ $# -lt 1 ]; then
    echo "Error: Missing required node list parameter"
    echo ""
    show_usage
    exit 1
fi

NODE_LIST="$1"
FF_THREADS="${2:-$DEFAULT_FF_THREADS}"
RECORDS_SIZE_M="${3:-$DEFAULT_RECORDS_SIZE_M}"
PAYLOAD_SIZE_B="${4:-$DEFAULT_PAYLOAD_SIZE_B}"
CSV_FILENAME="${5:-$DEFAULT_CSV_FILENAME}"

# Parse and validate node list
read -ra NODES_ARRAY <<< "$NODE_LIST"
if [ ${#NODES_ARRAY[@]} -eq 0 ]; then
    echo "Error: Empty node list provided"
    exit 1
fi

# Validate all node values are positive integers
for node in "${NODES_ARRAY[@]}"; do
    if ! [[ "$node" =~ ^[0-9]+$ ]] || [ "$node" -lt 1 ]; then
        echo "Error: Invalid node count '$node'. Must be positive integer."
        exit 1
    fi
done

# Validate other parameters
if ! [[ "$FF_THREADS" =~ ^[0-9]+$ ]] || [ "$FF_THREADS" -lt 1 ]; then
    echo "Error: FF_THREADS must be positive integer"
    exit 1
fi

if ! [[ "$RECORDS_SIZE_M" =~ ^[0-9]+$ ]] || [ "$RECORDS_SIZE_M" -lt 1 ]; then
    echo "Error: RECORDS_SIZE_M must be positive integer"
    exit 1
fi

if ! [[ "$PAYLOAD_SIZE_B" =~ ^[0-9]+$ ]] || [ "$PAYLOAD_SIZE_B" -lt 1 ]; then
    echo "Error: PAYLOAD_SIZE_B must be positive integer"
    exit 1
fi

echo "Scaling test: ${NODE_LIST} nodes, ${FF_THREADS} threads/proc, ${RECORDS_SIZE_M}M records"

# Clean up previous results
if [ -f "${CSV_FILENAME}" ]; then
    echo "Removing existing CSV file: ${CSV_FILENAME}"
    rm -f "${CSV_FILENAME}"
fi

# Initialize progress tracking
TOTAL_TESTS=${#NODES_ARRAY[@]}
CURRENT_TEST=0
START_TIME=$(date +%s)

echo "Starting tests at $(date)"
echo ""


# Configure FastFlow topology for baseline
srun --nodes=1 \
     --ntasks-per-node=1 \
     --time=00:02:00 \
     --mpi=pmix \
     bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh" >/dev/null 2>&1

# Run baseline test and capture output
BASELINE_OUTPUT=$(srun --ntasks=1 \
                       --nodes=1 \
                       --cpus-per-task=${FF_THREADS} \
                       --time=00:10:00 \
                       --mpi=pmix \
                       bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Baseline test failed"
    exit 1
fi

# Extract baseline time from CSV for speedup calculations
BASELINE_TIME=""
if [ -f "${CSV_FILENAME}" ]; then
    BASELINE_TIME=$(tail -n 1 "${CSV_FILENAME}" | cut -d',' -f6)
fi

echo ""
echo "MPI Procs   Time (ms)      Throughput (MRec/s)   Par Speedup  MPI Eff (%)  Total Eff (%)"
echo "----------- -------------- ------------------- ------------ ------------- --------------"

# Extract and display baseline with proper formatting
echo "$BASELINE_OUTPUT" | grep "^Mergesort FF" | sed 's/^Mergesort FF/1          /'

for nodes in "${NODES_ARRAY[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))

    # Skip baseline node if already processed
    if [ ${nodes} -eq 1 ]; then
        continue
    fi

    # Configure FastFlow topology
    srun --nodes=${nodes} \
         --ntasks-per-node=1 \
         --time=00:02:00 \
         --mpi=pmix \
         bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh" >/dev/null 2>&1

    # MPI scaling test
    srun --nodes=${nodes} \
         --ntasks=${nodes} \
         --ntasks-per-node=1 \
         --cpus-per-task=${FF_THREADS} \
         --time=00:10:00 \
         --mpi=pmix \
         bin/test_hybrid_performance ${FF_THREADS} ${RECORDS_SIZE_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet --skip-baselines --baseline-time=${BASELINE_TIME}

    # Exit on failure
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for ${nodes} nodes"
        exit 1
    fi
done

TOTAL_TIME=$(($(date +%s) - START_TIME))

echo ""
echo "Completed: ${TOTAL_TESTS} tests in $(printf '%d:%02d' $((TOTAL_TIME/60)) $((TOTAL_TIME%60)))"
echo "Results: ${CSV_FILENAME}"
