#!/bin/bash

## @file benchmark_cluster_weak_scaling.sh
## @brief Weak scaling test suite for hybrid MPI+FastFlow performance on cluster
## @details Maintains constant problem size per node while increasing node count
## @usage ./benchmark_cluster_weak_scaling.sh "1 2 4 8" [ff_threads] [records_per_node_m] [payload_b]
## @example ./benchmark_cluster_weak_scaling.sh "1 2 4 8"
## @example ./benchmark_cluster_weak_scaling.sh "1 2 4" 16 10 64

# ============================================================================
#                          CONFIGURATION PARAMETERS
# ============================================================================

# Default parameters (can be overridden via command line)
DEFAULT_FF_THREADS=10
DEFAULT_RECORDS_PER_NODE_M=10
DEFAULT_PAYLOAD_SIZE_B=16
DEFAULT_CSV_FILENAME="cluster_weak_scaling_results.csv"

# ============================================================================

show_usage() {
    cat << EOF
Usage: $0 "node_list" [ff_threads] [records_per_node_m] [payload_b] [csv_filename]

Parameters:
  node_list            Space-separated list of node counts (quoted)
  ff_threads           FastFlow threads per MPI process (default: ${DEFAULT_FF_THREADS})
  records_per_node_m   Dataset size per node in millions (default: ${DEFAULT_RECORDS_PER_NODE_M})
  payload_b            Payload size in bytes (default: ${DEFAULT_PAYLOAD_SIZE_B})
  csv_filename         Output CSV file (default: ${DEFAULT_CSV_FILENAME})

Examples:
  $0 "1 2 4 8"                     # 10M, 20M, 40M, 80M records
  $0 "1 2 4" 16 5 64               # 5M, 10M, 20M records with custom params
  $0 "1 2" 8 15 8 "weak.csv"       # 15M, 30M records, all custom

Weak Scaling Pattern:
  • 1 node:  1 × ${DEFAULT_RECORDS_PER_NODE_M}M = ${DEFAULT_RECORDS_PER_NODE_M}M records
  • 2 nodes: 2 × ${DEFAULT_RECORDS_PER_NODE_M}M = $((2 * DEFAULT_RECORDS_PER_NODE_M))M records
  • 4 nodes: 4 × ${DEFAULT_RECORDS_PER_NODE_M}M = $((4 * DEFAULT_RECORDS_PER_NODE_M))M records
  • etc.

Features:
  • Constant problem size per node (weak scaling)
  • Sequential srun execution (cluster-friendly)
  • Efficiency tracking vs single-node baseline
  • Progress tracking and error handling
  • Follows cluster guidelines (max 10min/job)

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
RECORDS_PER_NODE_M="${3:-$DEFAULT_RECORDS_PER_NODE_M}"
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

if ! [[ "$RECORDS_PER_NODE_M" =~ ^[0-9]+$ ]] || [ "$RECORDS_PER_NODE_M" -lt 1 ]; then
    echo "Error: RECORDS_PER_NODE_M must be positive integer"
    exit 1
fi

if ! [[ "$PAYLOAD_SIZE_B" =~ ^[0-9]+$ ]] || [ "$PAYLOAD_SIZE_B" -lt 1 ]; then
    echo "Error: PAYLOAD_SIZE_B must be positive integer"
    exit 1
fi

# ============================================================================
#                          EXECUTION SETUP
# ============================================================================

echo "=== Cluster Weak Scaling Test ==="
echo "Node configuration: ${NODE_LIST}"
echo "FastFlow threads/process: ${FF_THREADS}"
echo "Records per node: ${RECORDS_PER_NODE_M}M"
echo "Payload size: ${PAYLOAD_SIZE_B} bytes"
echo ""

# Display scaling pattern
echo "Weak Scaling Pattern:"
for node in "${NODES_ARRAY[@]}"; do
    total_records=$((node * RECORDS_PER_NODE_M))
    printf "  %d node(s): %d × %dM = %dM records\n" "$node" "$node" "$RECORDS_PER_NODE_M" "$total_records"
done
echo ""

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

# ============================================================================
#                          BASELINE ESTABLISHMENT (1 NODE)
# ============================================================================

# Get the first (smallest) node count for baseline
BASELINE_NODES=${NODES_ARRAY[0]}
BASELINE_RECORDS_M=$((BASELINE_NODES * RECORDS_PER_NODE_M))

echo "Establishing baseline with ${BASELINE_NODES} node(s), ${BASELINE_RECORDS_M}M records..."

# Configure FastFlow topology for baseline
srun --nodes=${BASELINE_NODES} \
     --ntasks-per-node=1 \
     --time=00:02:00 \
     --mpi=pmix \
     bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh" >/dev/null 2>&1

# Run baseline test
if [ ${BASELINE_NODES} -eq 1 ]; then
    # Single node: use test_hybrid_performance
    BASELINE_OUTPUT=$(srun --ntasks=1 \
                           --nodes=1 \
                           --cpus-per-task=${FF_THREADS} \
                           --time=00:10:00 \
                           --mpi=pmix \
                           bin/test_hybrid_performance ${FF_THREADS} ${BASELINE_RECORDS_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet 2>/dev/null)
else
    # Multi-node: use multi_node_main
    BASELINE_OUTPUT=$(srun --nodes=${BASELINE_NODES} \
                           --ntasks=${BASELINE_NODES} \
                           --ntasks-per-node=1 \
                           --cpus-per-task=${FF_THREADS} \
                           --time=00:10:00 \
                           --mpi=pmix \
                           bin/multi_node_main -s "${BASELINE_RECORDS_M}M" -r ${PAYLOAD_SIZE_B} -t ${FF_THREADS} --csv --filename ${CSV_FILENAME} 2>/dev/null | tail -1)
fi

if [ $? -ne 0 ]; then
    echo "Error: Baseline test failed"
    exit 1
fi

# Extract baseline time from output or CSV
BASELINE_TIME=""
if [ -f "${CSV_FILENAME}" ]; then
    BASELINE_TIME=$(tail -n 1 "${CSV_FILENAME}" | cut -d',' -f6)
fi

if [ -z "${BASELINE_TIME}" ]; then
    echo "Error: Could not extract baseline time"
    exit 1
fi

echo "Baseline established: ${BASELINE_TIME} ms"
echo ""

# ============================================================================
#                          RESULTS HEADER
# ============================================================================

echo "Nodes   Total Records   Time (ms)      Throughput (MRec/s)   Efficiency (%)"
echo "------- --------------- -------------- --------------------- --------------"

# Display baseline result
if [ ${BASELINE_NODES} -eq 1 ]; then
    echo "$BASELINE_OUTPUT" | grep "^Mergesort FF" | awk -v nodes="$BASELINE_NODES" -v records="${BASELINE_RECORDS_M}M" '{
        printf "%-7s %-15s %-14s %-21s 100.0\n", nodes, records, $3, $4
    }'
else
    # Parse multi-node output differently
    BASELINE_THROUGHPUT=$(echo "${BASELINE_OUTPUT}" | awk -F',' '{print $8}' 2>/dev/null || echo "N/A")
    printf "%-7s %-15s %-14s %-21s 100.0\n" "${BASELINE_NODES}" "${BASELINE_RECORDS_M}M" "${BASELINE_TIME}" "${BASELINE_THROUGHPUT}"
fi

# ============================================================================
#                          MAIN EXECUTION LOOP
# ============================================================================

for nodes in "${NODES_ARRAY[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))

    # Skip baseline if already processed
    if [ ${nodes} -eq ${BASELINE_NODES} ]; then
        continue
    fi

    TOTAL_RECORDS_M=$((nodes * RECORDS_PER_NODE_M))

    echo "Testing ${nodes} nodes (${TOTAL_RECORDS_M}M records)..." >&2

    # Configure FastFlow topology
    srun --nodes=${nodes} \
         --ntasks-per-node=1 \
         --time=00:02:00 \
         --mpi=pmix \
         bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh" >/dev/null 2>&1

    # Run scaling test
    if [ ${nodes} -eq 1 ]; then
        # Single node case
        TEST_OUTPUT=$(srun --ntasks=1 \
                           --nodes=1 \
                           --cpus-per-task=${FF_THREADS} \
                           --time=00:10:00 \
                           --mpi=pmix \
                           bin/test_hybrid_performance ${FF_THREADS} ${TOTAL_RECORDS_M} ${PAYLOAD_SIZE_B} ${CSV_FILENAME} --quiet --skip-baselines --baseline-time=${BASELINE_TIME} 2>/dev/null)
    else
        # Multi-node case
        TEST_OUTPUT=$(srun --nodes=${nodes} \
                           --ntasks=${nodes} \
                           --ntasks-per-node=1 \
                           --cpus-per-task=${FF_THREADS} \
                           --time=00:10:00 \
                           --mpi=pmix \
                           bin/multi_node_main -s "${TOTAL_RECORDS_M}M" -r ${PAYLOAD_SIZE_B} -t ${FF_THREADS} --csv --filename ${CSV_FILENAME} 2>/dev/null | tail -1)
    fi

    # Check for test failure
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for ${nodes} nodes" >&2
        exit 1
    fi

    # Extract and display results
    if [ ${nodes} -eq 1 ]; then
        # Parse single-node output
        echo "$TEST_OUTPUT" | grep "^Mergesort FF" | awk -v nodes="$nodes" -v records="${TOTAL_RECORDS_M}M" '{
            printf "%-7s %-15s %-14s %-21s %s\n", nodes, records, $3, $4, $7
        }'
    else
        # Parse multi-node CSV output
        if [ -f "${CSV_FILENAME}" ]; then
            LAST_LINE=$(tail -n 1 "${CSV_FILENAME}")
            TEST_TIME=$(echo "${LAST_LINE}" | cut -d',' -f6)
            TEST_THROUGHPUT=$(echo "${LAST_LINE}" | cut -d',' -f8)
            EFFICIENCY=$(echo "${LAST_LINE}" | cut -d',' -f10)

            printf "%-7s %-15s %-14s %-21s %.1f\n" "${nodes}" "${TOTAL_RECORDS_M}M" "${TEST_TIME}" "${TEST_THROUGHPUT}" "${EFFICIENCY}"
        else
            echo "Error: Could not find CSV results for ${nodes} nodes" >&2
        fi
    fi
done

# ============================================================================
#                          COMPLETION SUMMARY
# ============================================================================

TOTAL_TIME=$(($(date +%s) - START_TIME))

echo "" >&2
echo "Weak scaling test completed: ${TOTAL_TESTS} configurations in $(printf '%d:%02d' $((TOTAL_TIME/60)) $((TOTAL_TIME%60)))" >&2
echo "Results saved to: ${CSV_FILENAME}" >&2
echo "" >&2
echo "Weak Scaling Analysis:" >&2
echo "• Problem size per node: ${RECORDS_PER_NODE_M}M records" >&2
echo "• Ideal efficiency: 100% (constant time per node)" >&2
echo "• Efficiency drop indicates overhead from:" >&2
echo "  - Inter-node communication" >&2
echo "  - MPI coordination costs" >&2
echo "  - Load balancing issues" >&2
