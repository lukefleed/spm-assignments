#!/bin/bash

# Single-node array size scaling benchmark
# Measures performance across different dataset sizes with fixed thread count
# Usage: ./benchmark_array_scaling.sh [threads] [output_file]

set -e  # Exit on any error

THREADS=${1:-16}
OUTPUT=${2:-"benchmark_array_scaling_results.csv"}

SIZES=("200K" "400K" "800K" "1M" "2M" "4M" "8M" "16M" "32M" "64M" "100M")
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

# Remove existing output file to start fresh
rm -f "$OUTPUT"

# Initialize CSV with header by running first test
echo "Testing ${SIZES[0]}..."
./bin/single_node_main -s "${SIZES[0]}" -t "$THREADS" -r "$PAYLOAD_SIZE" --no-validate --csv --csv-file "$OUTPUT" --verbose

# Run remaining tests, appending to the same CSV file
for ((i=1; i<${#SIZES[@]}; i++)); do
    size="${SIZES[i]}"
    echo "Testing $size..."

    # Create temporary file for this test
    temp_file=$(mktemp)
    ./bin/single_node_main -s "$size" -t "$THREADS" -r "$PAYLOAD_SIZE" --no-validate --csv --csv-file "$temp_file" --verbose

    # Append data rows (skip header) to main output file
    tail -n +2 "$temp_file" >> "$OUTPUT"
    rm -f "$temp_file"
done

echo
echo "Benchmark completed successfully!"
echo "Results: $OUTPUT"
echo "CSV contains standardized performance metrics with comprehensive speedup analysis."
echo
echo "Sample output:"
echo "$(head -n 4 "$OUTPUT")"
echo "... ($(wc -l < "$OUTPUT") total records)"
