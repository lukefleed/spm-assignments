#!/bin/bash

# Single-node payload size scaling benchmark
# Measures performance across different record payload sizes with fixed array size
# Usage: ./benchmark_payload_scaling.sh [threads] [output_file]

set -e  # Exit on any error

THREADS=${1:-16}
OUTPUT=${2:-"benchmark_payload_scaling_results.csv"}
ARRAY_SIZE="10M"

PAYLOADS=(2 4 8 16 32 64 128 256 512)

if [ ! -f "./bin/single_node_main" ]; then
    echo "Error: ./bin/single_node_main not found"
    echo "Please build the project first with: make single_node_main"
    exit 1
fi

echo "=== Single Node Payload Size Scaling Benchmark ==="
echo "Thread count: $THREADS"
echo "Array size: $ARRAY_SIZE"
echo "Payload sizes: ${PAYLOADS[*]} bytes"
echo "Output file: $OUTPUT"
echo

# Remove existing output file to start fresh
rm -f "$OUTPUT"

# Initialize CSV with header by running first test
echo "Testing payload ${PAYLOADS[0]} bytes..."
./bin/single_node_main -s "$ARRAY_SIZE" -t "$THREADS" -r "${PAYLOADS[0]}" --no-validate --csv --csv-file "$OUTPUT" --verbose

# Run remaining tests, appending to the same CSV file
for ((i=1; i<${#PAYLOADS[@]}; i++)); do
    payload="${PAYLOADS[i]}"
    echo "Testing payload $payload bytes..."

    # Create temporary file for this test
    temp_file=$(mktemp)
    ./bin/single_node_main -s "$ARRAY_SIZE" -t "$THREADS" -r "$payload" --no-validate --csv --csv-file "$temp_file" --verbose

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
