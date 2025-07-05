#!/bin/bash

# Default parameters
DEFAULT_THREADS=8
DEFAULT_RECORDS_M=100
DEFAULT_PAYLOAD_B=16
DEFAULT_CSV_FILENAME="strong_scaling_results.csv"

show_usage() {
    cat << EOF
Usage: $0 "node_list" [threads] [records_m] [payload_b] [csv_filename]

Parameters:
  node_list     Space-separated list of node counts (e.g., "1 2 4 8")
  threads       Parallel threads per MPI process (default: ${DEFAULT_THREADS})
  records_m     Dataset size in millions of records (default: ${DEFAULT_RECORDS_M})
  payload_b     Payload size in bytes (default: ${DEFAULT_PAYLOAD_B})
  csv_filename  Output CSV file (default: ${DEFAULT_CSV_FILENAME})
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
RECORDS_M="${3:-$DEFAULT_RECORDS_M}"
PAYLOAD_B="${4:-$DEFAULT_PAYLOAD_B}"
CSV_FILENAME="${5:-$DEFAULT_CSV_FILENAME}"

# --- Execution ---
echo "=== Strong Scaling Test ==="
echo "Node counts:  ${NODE_LIST}"
echo "Threads/proc: ${THREADS}"
echo "Record count: ${RECORDS_M}M"
echo "Payload size: ${PAYLOAD_B}B"
echo ""

[ -f "${CSV_FILENAME}" ] && rm -f "${CSV_FILENAME}"
echo "MPI_Procs,Threads,Total_Time_ms,Speedup_vs_StdSort,Speedup_vs_Sequential,Speedup_vs_1Node" > "${CSV_FILENAME}"

# --- Run 1-Node Baseline ---
echo "Running 1-node baseline..."
BASELINE_OUTPUT=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=${THREADS} --time=00:10:00 --mpi=pmix bin/test_hybrid_performance ${THREADS} ${RECORDS_M} ${PAYLOAD_B})
if [ $? -ne 0 ]; then
    echo "Error: Baseline test failed." >&2
    exit 1
fi

IFS=',' read -r _ _ T_1NODE T_STDSORT T_SEQUENTIAL _ <<< "${BASELINE_OUTPUT}"

SPEEDUP_STD=$(awk "BEGIN {if($T_1NODE>0) printf \"%.2f\", $T_STDSORT/$T_1NODE; else print 0}")
SPEEDUP_SEQ=$(awk "BEGIN {if($T_1NODE>0) printf \"%.2f\", $T_SEQUENTIAL/$T_1NODE; else print 0}")

echo "1,${THREADS},${T_1NODE},${SPEEDUP_STD},${SPEEDUP_SEQ},1.00" >> "${CSV_FILENAME}"
echo "Baseline times captured. T_1NODE=${T_1NODE}ms, T_STDSORT=${T_STDSORT}ms, T_SEQUENTIAL=${T_SEQUENTIAL}ms"
echo ""

# --- Display Header ---
printf "%-11s %-14s %-20s %-24s %-18s\n" "MPI Procs" "Time (ms)" "Speedup vs StdSort" "Speedup vs Sequential" "Speedup vs 1-Node"
printf "%s\n" "--------------------------------------------------------------------------------------------"
printf "%-11d %-14.2f %-20.2f %-24.2f %-18.2f\n" 1 ${T_1NODE} ${SPEEDUP_STD} ${SPEEDUP_SEQ} 1.00

# --- Run Multi-Node Scaling Tests ---
read -ra NODES_ARRAY <<< "$NODE_LIST"
for nodes in "${NODES_ARRAY[@]}"; do
    if [ "$nodes" -eq 1 ]; then
        continue
    fi

    RUN_OUTPUT=$(srun --nodes=${nodes} --ntasks=${nodes} --ntasks-per-node=1 --cpus-per-task=${THREADS} --time=00:15:00 --mpi=pmix bin/test_hybrid_performance ${THREADS} ${RECORDS_M} ${PAYLOAD_B} \
        --t-stdsort "${T_STDSORT}" \
        --t-sequential "${T_SEQUENTIAL}" \
        --t-1node "${T_1NODE}")

    if [ $? -ne 0 ]; then
        echo "Error: Test failed for ${nodes} nodes." >&2
        exit 1
    fi

    IFS=',' read -r _ _ T_CURRENT _ _ _ <<< "${RUN_OUTPUT}"

    SPEEDUP_STD=$(awk "BEGIN {if($T_CURRENT>0) printf \"%.2f\", $T_STDSORT/$T_CURRENT; else print 0}")
    SPEEDUP_SEQ=$(awk "BEGIN {if($T_CURRENT>0) printf \"%.2f\", $T_SEQUENTIAL/$T_CURRENT; else print 0}")
    SPEEDUP_1NODE=$(awk "BEGIN {if($T_CURRENT>0) printf \"%.2f\", $T_1NODE/$T_CURRENT; else print 0}")

    echo "${nodes},${THREADS},${T_CURRENT},${SPEEDUP_STD},${SPEEDUP_SEQ},${SPEEDUP_1NODE}" >> "${CSV_FILENAME}"
    printf "%-11d %-14.2f %-20.2f %-24.2f %-18.2f\n" ${nodes} ${T_CURRENT} ${SPEEDUP_STD} ${SPEEDUP_SEQ} ${SPEEDUP_1NODE}
done

echo ""
echo "=== Strong Scaling Test Complete ==="
echo "Results saved to: ${CSV_FILENAME}"
