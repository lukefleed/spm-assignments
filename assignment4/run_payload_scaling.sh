#!/bin/bash

# Test single_node with fixed array size and variable payload
# Usage: ./run_payload_scaling.sh [threads] [output_file]

THREADS=${1:-8}
OUTPUT=${2:-"payload_scaling.csv"}
ARRAY_SIZE="10M"

PAYLOADS=(2 4 8 16 32 64 128 256 512)

if [ ! -f "./bin/single_node_main" ]; then
    echo "Error: ./bin/single_node_main not found"
    exit 1
fi

echo "Testing with $THREADS threads, array size $ARRAY_SIZE"
echo "Payloads: ${PAYLOADS[*]} bytes"
echo "Saving results to $OUTPUT"

echo "payload,threads,time_ms,throughput" > "$OUTPUT"

for payload in "${PAYLOADS[@]}"; do
    echo "Testing payload $payload bytes..."
    ./bin/single_node_main -s "$ARRAY_SIZE" -t "$THREADS" -r "$payload" --no-validate | tee -a "$OUTPUT"
done

echo "Done. Results saved to $OUTPUT"
