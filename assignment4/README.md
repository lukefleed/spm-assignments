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

**Purpose**: Interactive single-node parallel mergesort with performance comparison and CSV output capabilities.

```bash
./bin/single_node_main [OPTIONS]

Options:
  -s SIZE      Array size (e.g., 10M, 100M, 1G)
  -r BYTES     Record payload size in bytes (default: 8)
  -t THREADS   Number of FastFlow threads (default: 4)
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
Sequential MergeSort   3156.89      0.90x      ✓
FF Parallel MergeSort   891.45      3.19x      ✓
```

**CSV Output:**
When `--csv` is enabled, generates standardized CSV files with headers:

```csv
Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,Speedup_vs_Sequential,Efficiency_Percent,Valid
```

### 2. Multi-Node Application (`multi_node_main`)

**Purpose**: Distributed hybrid MPI+FastFlow mergesort with comprehensive performance metrics.

```bash
mpirun -np <processes> ./bin/multi_node_main [OPTIONS]

Options:
  -s SIZE      Array size (e.g., 10M, 100M)
  -r BYTES     Record payload size in bytes (default: 64)
  -t THREADS   FastFlow threads per MPI process (default: 4)
  -p PATTERN   Data pattern: random, sorted, reverse, nearly
  --no-validate        Skip correctness validation
  -v, --verbose        Enable verbose output
  -b, --benchmark      Run comprehensive benchmark suite
  -h, --help           Show help message
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

Comprehensive thread scaling analysis with dual-baseline comparison (std::sort and sequential mergesort). Generates both console output and CSV files for detailed analysis.

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

Dual-baseline performance analysis measuring both parallel and MPI scaling efficiency with comprehensive metrics.

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

**Output:** Creates detailed CSV with parallel baseline establishment and MPI scaling analysis.

#### Professional Benchmark Targets

For production-quality benchmarking with proper CSV output:

##### Single Configuration Benchmark

```bash
# Comprehensive single-configuration analysis
make benchmark_single_node ARRAY_SIZE_M=50 SINGLE_NODE_THREADS=8

# Custom CSV output
make benchmark_single_node SINGLE_NODE_OUTPUT=my_benchmark.csv
```

**Output:** `benchmark_single_node_results.csv` with detailed metrics including efficiency calculations.

##### Array Size Scaling Analysis

```bash
# Array size scaling (200K to 100M records)
make benchmark_array_scaling SINGLE_NODE_THREADS=4

# Custom thread count
make benchmark_array_scaling SINGLE_NODE_THREADS=8
```

**Output:** `benchmark_array_scaling_results.csv` with size-dependent performance characteristics.

##### Payload Size Scaling Analysis

```bash
# Payload size scaling (2 to 512 bytes)
make benchmark_payload_scaling SINGLE_NODE_THREADS=6

# Custom configuration
make benchmark_payload_scaling SINGLE_NODE_THREADS=4
```

**Output:** `benchmark_payload_scaling_results.csv` with memory hierarchy impact analysis.

### CSV Output Format

All benchmark targets generate standardized CSV files with the following format:

```csv
Test_Type,Implementation,Data_Size,Payload_Size_Bytes,Threads,Execution_Time_ms,Throughput_MRec_per_sec,Speedup_vs_StdSort,Speedup_vs_Sequential,Efficiency_Percent,Valid
```

**Key Metrics Explained:**

- **Speedup_vs_StdSort**: Performance relative to std::sort baseline
- **Speedup_vs_Sequential**: Performance relative to sequential mergesort
- **Efficiency_Percent**: Thread utilization efficiency (Speedup / Thread_Count \* 100)
- **Throughput_MRec_per_sec**: Million records processed per second

## Cluster Deployment

### Automated Benchmark Scripts

The project includes professional benchmark scripts for comprehensive performance analysis.

#### Array Size Scaling Analysis

Tests performance characteristics across different dataset sizes (200K to 100M records):

```bash
# Basic array scaling with default settings (16 threads)
./benchmark_array_scaling.sh

# Custom thread count and output file
./benchmark_array_scaling.sh 8 my_array_results.csv

# Via Makefile (recommended)
make benchmark_array_scaling SINGLE_NODE_THREADS=12
```

**Configuration:**

- **Dataset sizes:** 200K, 400K, 800K, 1M, 2M, 4M, 8M, 16M, 32M, 64M, 100M records
- **Fixed payload:** 16 bytes
- **Output:** Standardized CSV with size-dependent performance metrics

#### Payload Size Scaling Analysis

Tests memory hierarchy impact across different record sizes (2 to 512 bytes):

```bash
# Basic payload scaling with default settings (16 threads, 10M records)
./benchmark_payload_scaling.sh

# Custom configuration
./benchmark_payload_scaling.sh 8 my_payload_results.csv

# Via Makefile (recommended)
make benchmark_payload_scaling SINGLE_NODE_THREADS=6
```

**Configuration:**

- **Payload sizes:** 2, 4, 8, 16, 32, 64, 128, 256, 512 bytes
- **Fixed dataset:** 10M records
- **Output:** Standardized CSV with memory hierarchy analysis

#### Cluster MPI Scaling

For distributed cluster testing with the renamed professional script:

```bash
# Make script executable
chmod +x benchmark_cluster_scaling.sh

# Basic scaling test (1, 2, 4, 8 nodes)
./benchmark_cluster_scaling.sh "1 2 4 8"

# Custom configuration
./benchmark_cluster_scaling.sh "1 2 4" 16 50 64

# Full parameter specification
./benchmark_cluster_scaling.sh "1 2 4 8" 10 100 16 "scaling_results.csv"
```

**Script Parameters:**

```bash
./benchmark_cluster_scaling.sh "node_list" [ff_threads] [records_m] [payload_b] [csv_filename]
```

| Parameter      | Description                          | Default                          |
| -------------- | ------------------------------------ | -------------------------------- |
| `node_list`    | Space-separated node counts (quoted) | Required                         |
| `ff_threads`   | FastFlow threads per MPI process     | 10                               |
| `records_m`    | Dataset size in millions             | 100                              |
| `payload_b`    | Payload size in bytes                | 16                               |
| `csv_filename` | Output CSV file                      | `hybrid_performance_results.csv` |

### Batch Analysis Examples

**Comprehensive Thread Scaling Study:**

```bash
#!/bin/bash
echo "=== Thread Scaling Study ==="
for threads in 4 8 12 16 24; do
    echo "Testing $threads threads..."
    ./benchmark_array_scaling.sh $threads "array_scaling_${threads}t.csv"
    ./benchmark_payload_scaling.sh $threads "payload_scaling_${threads}t.csv"
done
```

**Multi-Configuration Analysis:**

```bash
#!/bin/bash
echo "=== Performance Matrix Analysis ==="

# Array scaling across different thread counts
for threads in 8 16 24; do
    make benchmark_array_scaling SINGLE_NODE_THREADS=$threads
    mv benchmark_array_scaling_results.csv "array_${threads}threads.csv"
done

# Payload scaling analysis
for threads in 8 16 24; do
    make benchmark_payload_scaling SINGLE_NODE_THREADS=$threads
    mv benchmark_payload_scaling_results.csv "payload_${threads}threads.csv"
done

echo "Analysis complete. Generated multiple CSV files for comparison."
```

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

## Command Reference

### Complete Makefile Targets

**Build Targets:**

- `make all` - Build all executables
- `make single_node_main` - Single-node application only
- `make multi_node_main` - Multi-node application only
- `make clean` - Clean build artifacts
- `make help` - Display comprehensive help

**Testing Targets:**

- `make test_correctness_single_node` - Single-node correctness verification
- `make test_correctness_hybrid` - Hybrid MPI correctness verification
- `make test_perf_single_node` - Thread scaling analysis with console output
- `make test_perf_hybrid` - MPI scaling analysis with dual baselines

**Benchmark Targets:**

- `make benchmark_single_node` - Single configuration comprehensive benchmark
- `make benchmark_array_scaling` - Array size scaling analysis (200K-100M)
- `make benchmark_payload_scaling` - Payload size scaling analysis (2-512B)

### Configuration Parameters

**Single-Node Parameters:**

- `THREAD_LIST="2 4 8"` - Thread counts for scaling tests
- `SINGLE_NODE_THREADS=8` - Single configuration thread count
- `ARRAY_SIZE_M=100` - Dataset size in millions
- `PAYLOAD_SIZE_B=16` - Record payload size in bytes
- `SINGLE_NODE_OUTPUT=file.csv` - CSV output filename

**Hybrid MPI Parameters:**

- `MPI_NODES_LIST="1 2 4 8"` - MPI process counts
- `FF_THREADS=4` - FastFlow threads per process
- `RECORDS_SIZE_M=10` - Dataset size in millions
- `PAYLOAD_SIZE_B=64` - Record payload size
- `HYBRID_CSV_FILE=results.csv` - CSV output filename

## Performance Analysis Guidelines

### Interpreting Results

**Speedup Metrics:**

- Values > 1.0 indicate performance improvement over baseline
- Linear scaling: speedup ≈ thread/process count
- Sublinear scaling: common due to synchronization overhead
- Superlinear scaling: rare, typically due to cache effects

**Efficiency Analysis:**

- Efficiency = (Speedup / Resource_Count) × 100%
- Values > 80% indicate excellent scaling
- Values 60-80% show good scaling
- Values < 60% suggest optimization opportunities

**Throughput Interpretation:**

- Higher throughput indicates better performance
- Compare across different configurations for scaling analysis
- Consider memory bandwidth limitations at high thread counts

### Best Practices

1. **Baseline Establishment:** Always run single-threaded baseline for accurate speedup calculation
2. **Multiple Runs:** For production analysis, average results across multiple runs
3. **Resource Monitoring:** Monitor CPU utilization and memory usage during benchmarks
4. **Configuration Documentation:** Document system specifications and compiler flags used
5. **Data Validation:** Enable correctness checking for critical benchmarks

## Troubleshooting

### Common Issues

**Build Problems:**

```bash
# Missing FastFlow
git submodule update --init --recursive

# Compilation errors
make clean && make all

# Missing MPI
# Install OpenMPI: sudo apt install libopenmpi-dev
```

**Runtime Issues:**

```bash
# Segmentation faults with large datasets
ulimit -s unlimited  # Increase stack size

# MPI communication failures
export OMPI_MCA_btl_base_warn_component_unused=0

# FastFlow topology issues
cd fastflow/ff && ./mapping_string.sh
```

**Performance Issues:**

- Ensure system has sufficient memory for large datasets
- Check CPU governor settings (performance vs powersave)
- Verify thread affinity and NUMA topology
- Monitor for thermal throttling during long benchmarks

## Project Structure

### Key Files and Directories

**Executables (bin/):**

- `single_node_main` - Interactive single-node mergesort with CSV output
- `multi_node_main` - Distributed hybrid MPI+FastFlow implementation
- `test_performance` - Thread scaling analysis tool
- `test_hybrid_performance` - MPI scaling analysis tool
- `test_correctness` - Single-node correctness verification
- `test_hybrid_correctness` - Multi-node correctness verification

**Benchmark Scripts:**

- `benchmark_array_scaling.sh` - Array size scaling (200K-100M records)
- `benchmark_payload_scaling.sh` - Payload size scaling (2-512 bytes)
- `benchmark_cluster_scaling.sh` - MPI cluster scaling analysis

**Core Libraries:**

- `include/csv_format.h` - Standardized CSV output format functions
- `src/common/utils.hpp` - Configuration parsing and utility functions
- `src/fastflow/ff_mergesort.hpp` - FastFlow parallel implementation
- `src/hybrid/mpi_ff_mergesort.hpp` - MPI+FastFlow hybrid implementation
- `src/sequential/sequential_mergesort.hpp` - Reference sequential implementation

**Configuration:**

- `Makefile` - Comprehensive build system with professional targets
- `fastflow/` - FastFlow parallel patterns library (git submodule)

### Output Files

**CSV Results:**

- `performance_results.csv` - Thread scaling test results
- `hybrid_performance_results.csv` - MPI scaling test results
- `benchmark_single_node_results.csv` - Single configuration benchmark
- `benchmark_array_scaling_results.csv` - Array size scaling results
- `benchmark_payload_scaling_results.csv` - Payload size scaling results

All CSV files follow the standardized format with dual-baseline speedup calculations and efficiency metrics for comprehensive performance analysis.

## License and Attribution

This implementation uses the FastFlow parallel patterns library for intra-node parallelization and OpenMPI for inter-node communication. Performance testing follows academic benchmarking best practices with proper statistical analysis and baseline comparisons.
