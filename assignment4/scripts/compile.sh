#!/bin/bash

echo "🔨 Compiling Hybrid MPI + FastFlow MergeSort..."

# Verifica FastFlow
if [ ! -d "./fastflow" ]; then
    echo "❌ Error: FastFlow directory not found!"
    exit 1
fi

echo "✅ FastFlow found at: ./fastflow"

# Verifica MPI
if ! command -v mpic++ &> /dev/null; then
    echo "❌ Error: mpic++ not found! Please install MPI."
    exit 1
fi

echo "✅ MPI compiler found: $(which mpic++)"

# Compilazione
echo "🔧 Building..."
make clean
make all

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "📦 Executables created:"
    echo "   • bin/single_node_sort"
    echo "   • bin/multi_node_sort"
    
    # Copia nella root per facilità
    make install
    echo "✅ Executables copied to project root"
    
    # Compila anche i test
    echo "🧪 Building tests..."
    make test
    if [ $? -eq 0 ]; then
        echo "✅ Tests compiled successfully"
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi

echo ""
echo "🚀 Ready to run:"
echo "   ./single_node_sort -s 1000000 -r 64 -t 4"
echo "   mpirun -np 2 ./multi_node_sort -s 1000000 -r 64 -t 4"
