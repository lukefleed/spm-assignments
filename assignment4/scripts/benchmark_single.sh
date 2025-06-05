#!/bin/bash

echo "📈 Single Node Performance Benchmark"

# Parametri per benchmark dettagliato
SIZE=10000000
PAYLOAD=64
THREADS=(1 2 4 8 16 32)

if [ ! -f "./single_node_sort" ]; then
    echo "❌ Error: single_node_sort not found. Compile first."
    exit 1
fi

echo "🔧 Benchmark Configuration:"
echo "   Array Size: $SIZE"
echo "   Record Payload: $PAYLOAD bytes"
echo "   Thread Counts: ${THREADS[*]}"
echo ""

echo "Threads,Time(ms),Speedup,Efficiency" > benchmark_single.csv

# Baseline sequenziale (1 thread)
echo "▶️  Running baseline (1 thread)..."
BASELINE_TIME=$(./single_node_sort -s $SIZE -r $PAYLOAD -t 1 | grep "Time:" | awk '{print $2}')

for threads in "${THREADS[@]}"; do
    echo "▶️  Testing with $threads threads..."
    TIME=$(./single_node_sort -s $SIZE -r $PAYLOAD -t $threads | grep "Time:" | awk '{print $2}')
    SPEEDUP=$(echo "scale=3; $BASELINE_TIME / $TIME" | bc -l)
    EFFICIENCY=$(echo "scale=3; $SPEEDUP / $threads" | bc -l)
    
    echo "$threads,$TIME,$SPEEDUP,$EFFICIENCY" >> benchmark_single.csv
    echo "   Time: ${TIME}ms, Speedup: ${SPEEDUP}x, Efficiency: ${EFFICIENCY}"
done

echo "✅ Benchmark completed! Results saved to benchmark_single.csv"
