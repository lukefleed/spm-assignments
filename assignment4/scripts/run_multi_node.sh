#!/bin/bash

echo "🚀 Running Multi Node MergeSort Tests..."

# Parametri di test
NODES=(1 2 4)
SIZES=(100000 1000000)
PAYLOADS=(64 256)
THREADS=(4 8)

# Verifica eseguibile
if [ ! -f "./multi_node_sort" ]; then
    echo "❌ Error: multi_node_sort not found. Please compile first."
    echo "Run: ./scripts/compile.sh"
    exit 1
fi

# Verifica MPI
if ! command -v mpirun &> /dev/null; then
    echo "❌ Error: mpirun not found!"
    exit 1
fi

echo "📊 Test Configuration:"
echo "   Nodes: ${NODES[*]}"
echo "   Sizes: ${SIZES[*]}"
echo "   Payloads: ${PAYLOADS[*]}"
echo "   Threads: ${THREADS[*]}"
echo ""

# Esecuzione test
for nodes in "${NODES[@]}"; do
    for size in "${SIZES[@]}"; do
        for payload in "${PAYLOADS[@]}"; do
            for threads in "${THREADS[@]}"; do
                echo "▶️  Testing: nodes=$nodes, size=$size, payload=$payload, threads=$threads"
                mpirun -np $nodes ./multi_node_sort -s $size -r $payload -t $threads
                echo "---"
            done
        done
    done
done

echo "✅ Multi node tests completed!"
