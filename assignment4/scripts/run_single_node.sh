#!/bin/bash

echo "🚀 Running Single Node MergeSort Tests..."

# Parametri di test
SIZES=(100000 1000000)
PAYLOADS=(8 64 256)
THREADS=(1 2 4 8)

# Verifica eseguibile
if [ ! -f "./single_node_sort" ]; then
    echo "❌ Error: single_node_sort not found. Please compile first."
    echo "Run: ./scripts/compile.sh"
    exit 1
fi

echo "📊 Test Configuration:"
echo "   Sizes: ${SIZES[*]}"
echo "   Payloads: ${PAYLOADS[*]}"
echo "   Threads: ${THREADS[*]}"
echo ""

# Esecuzione test
for size in "${SIZES[@]}"; do
    for payload in "${PAYLOADS[@]}"; do
        for threads in "${THREADS[@]}"; do
            echo "▶️  Testing: size=$size, payload=$payload, threads=$threads"
            ./single_node_sort -s $size -r $payload -t $threads
            echo "---"
        done
    done
done

echo "✅ Single node tests completed!"
