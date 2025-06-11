#!/bin/bash

# Single-node array size scaling benchmark
# Measures performance across different dataset sizes with fixed thread count
# Usage: ./benchmark_array_scaling.sh [threads] [output_file]

set -e  # Exit on any error

THREADS=${1:-16}
OUTPUT=${2:-"benchmark_array_scaling_results.csv"}

SIZES=("200K" "400K" "800K" "1M" "2M" "4M" "8M" "16M" "32M" "64M")
PAYLOAD_SIZE=16

if [ ! -f "./bin/single_node_main" ]; then
    echo "Error: ./bin/single_node_main not found"
    echo "Please build the project first with: make single_node_main"
    exit 1
fi

echo "=== Single Node Array Size Scaling Benchmark ==="
echo "Thread count: $THREADS"
echo "Payload size: $PAYLOAD_SIZE bytes"
echo "Array sizes: ${SIZES[*]}"
echo "Output file: $OUTPUT"
echo

# Create CSV with header
rm -f "$OUTPUT"
echo "Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,Speedup_vs_Sequential,Efficiency_Percent,Valid" > "$OUTPUT"

for size in "${SIZES[@]}"; do
    echo "Testing $size..."

    # Run test (CSV will be written to timestamped file)
    ./bin/single_node_main -s "$size" -t "$THREADS" -r "$PAYLOAD_SIZE" --csv --verbose

    # Find the most recently created CSV file and append its data (skip header)
    LATEST_CSV=$(ls -t results_single_node_*.csv | head -1)
    if [ -f "$LATEST_CSV" ]; then
        tail -n +2 "$LATEST_CSV" >> "$OUTPUT"
        rm -f "$LATEST_CSV"  # Clean up temporary file
    else
        echo "Error: Could not find generated CSV file for size $size"
        exit 1
    fi
done

echo
echo "Benchmark completed successfully!"
echo "Results: $OUTPUT"
echo "CSV contains standardized performance metrics with comprehensive speedup analysis."
echo
echo "Sample output:"
echo "$(head -n 4 "$OUTPUT")"
echo "... ($(tail -n +2 "$OUTPUT" | wc -l) data records)"
