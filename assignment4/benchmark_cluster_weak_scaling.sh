#!/bin/bash

# Default parameters
DEFAULT_THREADS=8
DEFAULT_RECORDS_PER_NODE_M=10
DEFAULT_PAYLOAD_B=16
DEFAULT_CSV_FILENAME="weak_scaling_results.csv"

show_usage() {
    cat << EOF
Usage: $0 "node_list" [threads] [records_per_node_m] [payload_b] [csv_filename]

Parameters:
  node_list          Space-separated list of node counts (e.g., "1 2 4 8")
  threads            Parallel threads per MPI process (default: ${DEFAULT_THREADS})
  records_per_node_m Dataset size per node in millions (default: ${DEFAULT_RECORDS_PER_NODE_M})
  payload_b          Payload size in bytes (default: ${DEFAULT_PAYLOAD_B})
  csv_filename       Output CSV file (default: ${DEFAULT_CSV_FILENAME})
EOF
}

# --- Argument Parsing ---
if [ $# -lt 1 ]; then
    echo "Error: Missing required node list." >&2
    show_usage
    exit 1
fi

NODE_LIST="$1"
THREADS="${2:-$DEFAULT_THREADS}"
RECORDS_PER_NODE_M="${3:-$DEFAULT_RECORDS_PER_NODE_M}"
PAYLOAD_B="${4:-$DEFAULT_PAYLOAD_B}"
CSV_FILENAME="${5:-$DEFAULT_CSV_FILENAME}"

# --- Execution ---
echo "=== Weak Scaling Test ==="
echo "Node counts:      ${NODE_LIST}"
echo "Threads/proc:     ${THREADS}"
echo "Records/node:     ${RECORDS_PER_NODE_M}M"
echo "Payload size:     ${PAYLOAD_B}B"
echo ""

[ -f "${CSV_FILENAME}" ] && rm -f "${CSV_FILENAME}"
echo "MPI_Procs,Threads,Total_Records_M,Time_ms,Efficiency_%" > "${CSV_FILENAME}"

printf "%-11s %-19s %-14s %-14s\n" "MPI Procs" "Total Records (M)" "Time (ms)" "Efficiency (%)"
printf "%s\n" "----------------------------------------------------------------"

read -ra NODES_ARRAY <<< "$NODE_LIST"
T_BASELINE=""  # Will store the baseline time (1 node)

for nodes in "${NODES_ARRAY[@]}"; do
    TOTAL_RECORDS_M=$((nodes * RECORDS_PER_NODE_M))

    # We only need the timing for the hybrid run itself.
    RUN_OUTPUT=$(srun --nodes=${nodes} --ntasks=${nodes} --ntasks-per-node=1 --cpus-per-task=${THREADS} --time=00:15:00 --mpi=pmix bin/test_hybrid_performance ${THREADS} ${TOTAL_RECORDS_M} ${PAYLOAD_B})
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for ${nodes} nodes." >&2
        exit 1
    fi

    IFS=',' read -r _ _ T_CURRENT _ _ _ <<< "${RUN_OUTPUT}"

    # Store baseline time (first node count, assumed to be 1)
    if [ -z "${T_BASELINE}" ]; then
        T_BASELINE="${T_CURRENT}"
    fi

    # Calculate efficiency (speedup/nodes)
    SPEEDUP=$(echo "scale=4; ${T_BASELINE} / ${T_CURRENT}" | bc -l)
    EFFICIENCY_PERCENT=$(echo "scale=2; (${SPEEDUP} / ${nodes}) * 100" | bc -l)

    echo "${nodes},${THREADS},${TOTAL_RECORDS_M},${T_CURRENT},${EFFICIENCY_PERCENT}" >> "${CSV_FILENAME}"
    printf "%-11d %-19d %-14.2f %-14.2f%%\n" ${nodes} ${TOTAL_RECORDS_M} ${T_CURRENT} ${EFFICIENCY_PERCENT}
done

echo ""
echo "=== Weak Scaling Test Complete ==="
echo "Results saved to: ${CSV_FILENAME}"
