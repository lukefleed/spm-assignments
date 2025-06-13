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

**Purpose**: Interactive single-node parallel mergesort.

```bash
./bin/single_node_main [OPTIONS]

Options:
  -s SIZE            Array size (e.g., 10M, 100M, 1G)
  -r BYTES           Record payload size in bytes (default: 8)
  -t THREADS         Number of FastFlow threads (default: 4)
  --pattern PATTERN  Data pattern: random, sorted, reverse, nearly (default: random)
  --no-validate      Skip correctness validation for faster benchmarking
  --csv              Enable CSV output mode
  --csv-file FILE    Specify CSV output filename
  -v, --verbose      Show progress when in CSV mode
```

**Example Usage:**

```bash
# Default interactive test: 1M records, 8B payload, 4 threads
./bin/single_node_main

# Large test with custom configuration
./bin/single_node_main -s 50M -r 64 -t 8 --pattern random

# CSV benchmark mode with progress output
./bin/single_node_main -s 10M -t 16 --csv --csv-file results.csv --verbose

# Fast benchmark without validation
./bin/single_node_main -s 100M -t 12 --no-validate --csv
```

**Sample Console Output:**

```
=== Single Node MergeSort Comparison ===
Array size: 10000000 elements
Payload size: 64 bytes
Total data: 640.00 MB
Threads: 8
Pattern: Random

Implementation      Time (ms)    Speedup    Valid
----------------------------------------------------
std::sort              2847.23      1.00x      ✓
Sequential MergeSort   3156.89      0.40x      ✓
FF Parallel MergeSort   891.45      2.19x      ✓
```

**CSV Output:**
When `--csv` is enabled, generates standardized CSV files with headers:

```csv
Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,Speedup_vs_Sequential,Efficiency_Percent,Valid
```

### 2. Multi-Node Application (`multi_node_main`)

**Purpose**: Distributed hybrid MPI+FastFlow mergesort.

```bash
mpirun -np <processes> ./bin/multi_node_main [OPTIONS]

Options:
  -s SIZE            Array size (e.g., 10M, 100M)
  -r BYTES           Record payload size in bytes (default: 64)
  -t THREADS         Number of parallel threads per node (default: 4)
  -p PATTERN         Data pattern: random, sorted, reverse, nearly
  --no-validate      Disable result validation
  --verbose          Enable verbose output
  --benchmark        Enable benchmark mode
  --help             Show this help message
```

**Example Usage:**

```bash
# Basic 4-process test
mpirun -np 4 ./bin/multi_node_main -s 1M -t 8

# Benchmark mode with detailed analysis
mpirun -np 8 ./bin/multi_node_main --benchmark -s 50M -r 16

# Large payload test with verbose output
mpirun -np 2 ./bin/multi_node_main -s 1M -r 256 --verbose
```

**Sample Output:**

```
=== Multi-Node Hybrid MPI+Parallel MergeSort Results ===
Problem Configuration:
  Array size: 100000000 elements
  Payload size: 64 bytes
  Total data: 6.40 GB
  MPI processes: 4
  Parallel threads per node: 8
  Data pattern: Random

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

### Correctness Verification

#### Single-Node Correctness (`test_correctness`)

Validates both sequential and parallel implementations across various test cases.

```bash
# Via makefile (recommended)
make test_correctness_single_node

# Manual execution
./bin/test_correctness
```

**Test Coverage:**

- Edge cases (empty, single element, two elements)
- Small datasets with different thread counts
- Medium and large-scale stress tests
- Various payload sizes (0B to 256B)
- Thread scalability (1 to 16+ threads)
- Special cases (power-of-2 sizes, prime sizes)

#### Hybrid Correctness (`test_hybrid_correctness`)

Validates distributed sorting across multiple MPI processes.

```bash
# Via makefile (recommended)
make test_correctness_hybrid

# Manual execution (2 processes)
mpirun -np 2 ./bin/test_hybrid_correctness
```

**Test Coverage:**

- Multi-process data distribution
- Various payload configurations
- Different data patterns
- Thread scaling per process
- Element preservation validation

### Performance Testing

#### Single-Node Performance Analysis (`test_perf_single_node`)

Comprehensive thread scaling analysis with dual-baseline comparison (std::sort and sequential mergesort). Generates both console output and CSV files.

```bash
# Basic thread scaling test (default configuration)
make test_perf_single_node

# Custom thread scaling
make test_perf_single_node THREAD_LIST="2 4 8 16" ARRAY_SIZE_M=20 PAYLOAD_SIZE_B=128

# Quick test with fewer threads
make test_perf_single_node THREAD_LIST="1 2 4" ARRAY_SIZE_M=5 PAYLOAD_SIZE_B=32
```

**Available Parameters:**

- `THREAD_LIST`: Space-separated thread counts (default: "2 4 6 8 10 12 24")
- `ARRAY_SIZE_M`: Dataset size in millions (default: 10)
- `PAYLOAD_SIZE_B`: Payload size in bytes (default: 64)

**Output:** Creates `performance_results.csv` with standardized format including speedup vs both std::sort and sequential baselines.

**Manual Usage:**

```bash
# Custom thread scaling: 1-16 threads, 10M records, 64B payload
./bin/test_performance "1 2 4 8 16" 10 64
```

#### Hybrid MPI Performance Analysis (`test_perf_hybrid`)

Dual-baseline performance analysis measuring both parallel and MPI scaling efficiency.

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

**Output:** CSV file `hybrid_performance_results.csv`

#### Array Size Scaling Analysis

```bash
# Array size scaling (200K to 100M records)
make benchmark_array_scaling SINGLE_NODE_THREADS=4

# Custom thread count
make benchmark_array_scaling SINGLE_NODE_THREADS=8
```

**Output:** `benchmark_array_scaling_results.csv`

#### Payload Size Scaling Analysis

```bash
# Payload size scaling (2 to 512 bytes)
make benchmark_payload_scaling SINGLE_NODE_THREADS=6
```

**Output:** `benchmark_payload_scaling_results.csv`

### Cluster MPI Scaling

For distributed cluster testing with strong and weak scaling analysis:

**Strong Scaling** (fixed problem size, increasing nodes):

```bash
# Make script executable
chmod +x benchmark_cluster_scaling.sh

# Basic scaling test (1, 2, 4, 8 nodes)
./benchmark_cluster_scaling.sh "1 2 4 8"

# Custom configuration
./benchmark_cluster_scaling.sh "1 2 4" 16 100 64

# Full parameter specification
./benchmark_cluster_scaling.sh "1 2 4 8" 10 100 16 "strong_scaling_results.csv"
```

**Weak Scaling** (problem size per node constant):

```bash
# Make script executable
chmod +x benchmark_cluster_weak_scaling.sh

# Basic weak scaling test
./benchmark_cluster_weak_scaling.sh "1 2 4 8"

# Custom configuration (10M records per node)
./benchmark_cluster_weak_scaling.sh "1 2 4" 16 10 64

# Full parameter specification
./benchmark_cluster_weak_scaling.sh "1 2 4 8" 10 10 16 "weak_scaling_results.csv"
```

**Strong Scaling Script Parameters:**

```bash
./benchmark_cluster_scaling.sh "node_list" [ff_threads] [records_m] [payload_b] [csv_filename]
```

| Parameter      | Description                          | Default                              |
| -------------- | ------------------------------------ | ------------------------------------ |
| `node_list`    | Space-separated node counts (quoted) | Required                             |
| `ff_threads`   | FastFlow threads per MPI process     | 10                                   |
| `records_m`    | Total dataset size in millions       | 100                                  |
| `payload_b`    | Payload size in bytes                | 16                                   |
| `csv_filename` | Output CSV file                      | `cluster_strong_scaling_results.csv` |

**Weak Scaling Script Parameters:**

```bash
./benchmark_cluster_weak_scaling.sh "node_list" [ff_threads] [records_per_node_m] [payload_b] [csv_filename]
```

| Parameter            | Description                          | Default                            |
| -------------------- | ------------------------------------ | ---------------------------------- |
| `node_list`          | Space-separated node counts (quoted) | Required                           |
| `ff_threads`         | FastFlow threads per MPI process     | 10                                 |
| `records_per_node_m` | Dataset size per node in millions    | 10                                 |
| `payload_b`          | Payload size in bytes                | 16                                 |
| `csv_filename`       | Output CSV file                      | `cluster_weak_scaling_results.csv` |

### Manual SLURM Commands

For direct SLURM control without automated scripts:

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
