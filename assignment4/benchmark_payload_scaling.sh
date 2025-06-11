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

# Create consolidated CSV with header
rm -f "$OUTPUT"
echo "Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,Speedup_vs_Sequential,Efficiency_Percent,Valid" > "$OUTPUT"

for payload in "${PAYLOADS[@]}"; do
    echo "Testing payload $payload bytes..."

    # Create a temporary log file to capture output while showing it
    TEMP_LOG=$(mktemp)

    # Run test, show output in real-time, and capture it for CSV filename extraction
    ./bin/single_node_main -s "$ARRAY_SIZE" -t "$THREADS" -r "$payload" --no-validate --csv --verbose 2>&1 | tee "$TEMP_LOG"

    # Extract the CSV filename from the captured output
    CSV_FILE=$(grep "CSV output will be written to:" "$TEMP_LOG" | sed 's/.*CSV output will be written to: //')

    # Clean up temp log
    rm -f "$TEMP_LOG"

    # Wait a moment for file to be fully written
    sleep 0.2

    # Append CSV data from the generated file (skip header line)
    if [ -f "$CSV_FILE" ] && [ -s "$CSV_FILE" ]; then
        tail -n +2 "$CSV_FILE" >> "$OUTPUT"
        rm -f "$CSV_FILE"  # Clean up the temporary file
    else
        echo "Warning: Expected CSV file $CSV_FILE not found or empty"
    fi

    echo # Add blank line between tests
done

# Clean up any remaining CSV files that might have been left behind
echo "Cleaning up temporary files..."
rm -f results_single_node_*.csv

echo
echo "Benchmark completed successfully!"
echo "Results: $OUTPUT"
