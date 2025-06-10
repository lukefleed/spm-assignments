#!/bin/bash

# Test single_node with fixed threads and variable array sizes
# Usage: ./run_single_node_scaling.sh [threads] [output_file]

THREADS=${1:-8}
OUTPUT=${2:-"array_size_scaling.csv"}

SIZES=("200K" "400K" "800K" "1M" "2M" "4M" "8M" "16M" "32M" "64M" "100M")

if [ ! -f "./bin/single_node_main" ]; then
    echo "Error: ./bin/single_node_main not found"
    exit 1
fi

echo "Testing with $THREADS threads"
echo "Array sizes: ${SIZES[*]}"
echo "Saving results to $OUTPUT"

echo "size,threads,time_ms,throughput" > "$OUTPUT"

for size in "${SIZES[@]}"; do
    echo "Testing $size..."
    ./bin/single_node_main -s "$size" -t "$THREADS" -r 64 --no-validate | tee -a "$OUTPUT"
done

echo "Done. Results saved to $OUTPUT"
