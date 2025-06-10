# Hybrid MPI+FastFlow MergeSort - User Documentation

## Overview

This project implements a scalable distributed mergesort algorithm combining **MPI inter-node communication** with **FastFlow intra-node parallelization**. The implementation provides both single-node parallel sorting and multi-node hybrid distributed sorting capabilities.

## Quick Start

```bash
# Build all executables
make all

# Run single-node correctness tests
make test_correctness_single_node

# Run hybrid correctness tests
make test_correctness_hybrid

# Run single-node performance analysis
make test_perf_single_node

# Run hybrid performance analysis
make test_perf_hybrid
```

## Build System

### Basic Commands

```bash
# Build all targets
make all

# Clean build artifacts
make clean

# Display help with all options
make help
```

## Executables Usage

### 1. Single-Node Application (`single_node_main`)

**Purpose**: Interactive single-node parallel mergesort with performance comparison.

```bash
./bin/single_node_main [OPTIONS]

Options:
  -s SIZE     Array size (e.g., 10M, 100M, 1G)
  -r BYTES    Record payload size in bytes (default: 8)
  -t THREADS  Number of FastFlow threads (default: 4)
  --pattern PATTERN  Data pattern: random, sorted, reverse, nearly (default: random)
```

**Example Usage:**

```bash
# Default test: 1M records, 8B payload, 4 threads
./bin/single_node_main

# Large test with custom configuration
./bin/single_node_main -s 50M -r 64 -t 8 --pattern random
```

**Sample Output:**

```
=== Single Node MergeSort Comparison ===
Implementation          Time (ms)    Speedup    Valid
----------------------------------------------------
std::sort               2847.23      1.00x      ✓
Sequential MergeSort    3156.89      0.90x      ✓
FF Parallel MergeSort   891.45       3.19x      ✓
```

### 2. Multi-Node Application (`multi_node_main`)

**Purpose**: Distributed hybrid MPI+FastFlow mergesort with comprehensive metrics.

```bash
mpirun -np <processes> ./bin/multi_node_main [OPTIONS]

Options:
  -s SIZE     Array size (e.g., 10M, 100M)
  -r BYTES    Record payload size in bytes (default: 64)
  -t THREADS  FastFlow threads per MPI process (default: 4)
  -p PATTERN  Data pattern: random, sorted, reverse, nearly
  --benchmark         Run comprehensive benchmark suite (for testing)
```

**Example Usage:**

```bash
# Basic 4-process test
mpirun -np 4 ./bin/multi_node_main -s 1M -t 8

# Benchmark mode with detailed analysis
mpirun -np 8 ./bin/multi_node_main --benchmark -s 50M -r 16

# Large payload test
mpirun -np 2 ./bin/multi_node_main -s 1M -r 256
```

**Sample Output:**

```
=== Multi-Node Hybrid MPI+Parallel MergeSort Results ===
Problem Configuration:
  Array size: 100000000 elements
  Payload size: 64 bytes
  MPI processes: 4
  Parallel threads per node: 8

Performance Results:
  Total execution time: 2156.78 ms
  Local sort time: 1245.32 ms
  Communication time: 423.67 ms
  Merge time: 487.79 ms
  Communication ratio: 19.6%
  Computation ratio: 57.7%
  Throughput: 46.37 M elements/sec
```

## Testing Framework

### 3. Correctness Tests

#### Single-Node Correctness (`test_correctness`)

**Purpose**: Validates both sequential and parallel implementations across various test cases.

```bash
# Manual execution
./bin/test_correctness

# Via makefile (recommended)
make test_correctness_single_node
```

**Test Coverage:**

- Edge cases (empty, single element, two elements)
- Small datasets with different thread counts
- Medium and large-scale stress tests
- Various payload sizes (0B to 256B)
- Thread scalability (1 to 16+ threads)
- Special cases (power-of-2 sizes, prime sizes)

#### Hybrid Correctness (`test_hybrid_correctness`)

**Purpose**: Validates distributed sorting across multiple MPI processes.

```bash
# Manual execution (2 processes)
mpirun -np 2 ./bin/test_hybrid_correctness

# Via makefile (recommended)
make test_correctness_hybrid
```

**Test Coverage:**

- Multi-process data distribution
- Various payload configurations
- Different data patterns
- Thread scaling per process
- Element preservation validation

### 4. Performance Benchmarking

#### Single-Node Performance (`test_performance`)

**Purpose**: Comprehensive performance analysis across implementations, thread counts, and problem sizes.

```bash
# Basic thread scaling test
make test_perf_single_node

# Custom configuration
make test_perf_single_node THREAD_LIST="2 4 8 16" ARRAY_SIZE_M=20 PAYLOAD_SIZE_B=128

# With additional scaling tests
make test_perf_single_node EXTRA_FLAGS="--size-scaling --payload-scaling"
```

**Available Parameters:**

- `THREAD_LIST`: Space-separated thread counts (default: "2 4 6 8 10 12 24")
- `ARRAY_SIZE_M`: Dataset size in millions (default: 10)
- `PAYLOAD_SIZE_B`: Payload size in bytes (default: 64)
- `EXTRA_FLAGS`: Additional options (`--size-scaling`, `--payload-scaling`)

**Manual Usage:**

```bash
# Thread scaling: 1-16 threads, 10M records, 64B payload
./bin/test_performance "1 2 4 8 16" 10 64

# Size scaling test
./bin/test_performance "8" 5 64 --size-scaling

# Complete analysis
./bin/test_performance "2 4 8 12" 20 128 --size-scaling --payload-scaling
```

#### Hybrid Performance (`test_hybrid_performance`)

**Purpose**: Dual-baseline performance analysis measuring both parallel and MPI scaling efficiency.

```bash
# Basic hybrid performance test
make test_perf_hybrid

# Custom MPI scaling test
make test_perf_hybrid MPI_NODES_LIST="1 2 4 8" FF_THREADS=8

# Large-scale test
make test_perf_hybrid RECORDS_SIZE_M=100 PAYLOAD_SIZE_B=256 FF_THREADS=12
```

**Available Parameters:**

- `MPI_NODES_LIST`: MPI process counts (default: "1 2 4 8")
- `FF_THREADS`: FastFlow threads per process (default: 4)
- `RECORDS_SIZE_M`: Dataset size in millions (default: 10)
- `PAYLOAD_SIZE_B`: Payload size in bytes (default: 64)
- `HYBRID_CSV_FILE`: Output CSV filename (default: hybrid_performance_results.csv)

**Manual Usage:**

```bash
# Single test with 4 processes, 8 threads each
mpirun -np 4 ./bin/test_hybrid_performance 8 50 128 results.csv

# Scaling analysis
mpirun -np 1 ./bin/test_hybrid_performance 4 10 64 baseline.csv
mpirun -np 4 ./bin/test_hybrid_performance 4 10 64 scaling.csv
```

#### Sequential Baseline (`test_sequential`)

**Purpose**: Simple comparison between custom sequential mergesort and std::sort.

```bash
./bin/test_sequential -s 10M -r 64 --pattern random
```

### Performance Metrics Explanation

- **Parallel Speedup**: Performance vs single-node parallel baseline
- **MPI Efficiency**: How well MPI processes scale (Speedup / MPI Processes)
- **Total Efficiency**: Overall scaling efficiency (Speedup / Total Threads)
- **Throughput**: Million records or MB processed per second
- **Communication Ratio**: Percentage of time spent in MPI communication

## Cluster Deployment (SLURM)

### Automated Cluster Scaling Script

The project includes `run_cluster_scaling.sh` for automated cluster scaling tests following SLURM best practices.

#### Basic Usage

```bash
# Make script executable
chmod +x run_cluster_scaling.sh

# Basic scaling test (1, 2, 4, 8 nodes)
./run_cluster_scaling.sh "1 2 4 8"

# Custom configuration
./run_cluster_scaling.sh "1 2 4" 16 50 64

# Full parameter specification
./run_cluster_scaling.sh "1 2 4 8" 10 100 16 "scaling_results.csv"
```

#### Script Parameters

```bash
./run_cluster_scaling.sh "node_list" [ff_threads] [records_m] [payload_b] [csv_filename]
```

| Parameter      | Description                          | Default                          |
| -------------- | ------------------------------------ | -------------------------------- |
| `node_list`    | Space-separated node counts (quoted) | Required                         |
| `ff_threads`   | FastFlow threads per MPI process     | 10                               |
| `records_m`    | Dataset size in millions             | 100                              |
| `payload_b`    | Payload size in bytes                | 16                               |
| `csv_filename` | Output CSV file                      | `hybrid_performance_results.csv` |

**Batch Processing Multiple Configurations:**

```bash
#!/bin/bash
# Run multiple scaling studies

echo "=== Thread Scaling Study ==="
for threads in 8 16 24 32; do
    echo "Testing $threads threads..."
    ./run_cluster_scaling.sh "1 2 4 8" $threads 100 64 "scaling_${threads}t.csv"
done

echo "=== Payload Scaling Study ==="
for payload in 16 64 256 1024; do
    echo "Testing ${payload}B payload..."
    ./run_cluster_scaling.sh "1 2 4 8" 16 100 $payload "scaling_${payload}b.csv"
done
```

### Manual SLURM Commands

For direct SLURM control without the automated script:

```bash
# Single test
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=12 --time=00:10:00 --mpi=pmix \
     bin/multi_node_main -s 100M -t 12 --benchmark

# Performance test
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 --time=00:15:00 --mpi=pmix \
     bin/test_hybrid_performance 16 200 64 manual_results.csv

# FastFlow topology configuration (run once per node allocation)
srun --nodes=4 --ntasks-per-node=1 --time=00:02:00 --mpi=pmix \
     bash -c "cd fastflow/ff && echo 'y' | ./mapping_string.sh"
```
