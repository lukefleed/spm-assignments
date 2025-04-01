## Project Overview

The project implements and benchmarks different scheduling algorithms for calculating the maximum number of steps in the Collatz sequence (also known as the 3n+1 problem) for ranges of numbers. It compares:

- Sequential execution (baseline)
- Static scheduling (Block, Cyclic, and Block-Cyclic variants)
- Dynamic scheduling (task queue-based)

## Building the Project

To build the project, simply run:

```bash
make
```

This will compile all source files and create the `collatz_par` executable in the build directory.

## Running Tests

### Correctness Testing

To verify that all parallel implementations produce correct results compared to the sequential version:

```bash
make test_correctness
# or simply
make test
```

This runs a comprehensive suite of test cases including edge cases (empty ranges, single values, thread oversubscription) to ensure all scheduling algorithms produce identical results.

### Performance Benchmarking

To run the full performance benchmark suite:

```bash
make benchmark
```

This executes multiple configurations across:

- Different workloads (balanced, unbalanced, many small ranges, etc.)
- Thread counts (2 to maximum hardware concurrency)
- Chunk sizes (16, 32, 64, 128, 256, 512, 1024)
- All scheduler types

Results are saved to `performance_results.csv`.

## Benchmark Design

### What We Test

1. **Different Workloads**:

   - Medium balanced (1-100k)
   - Large balanced (1-1M)
   - Unbalanced mix of small, medium, and large ranges
   - Many small uniform ranges
   - Ranges around powers of 2 (2^8 to 2^20)
   - Extreme imbalance with isolated expensive calculations

2. **Scheduling Strategies**:

   - Sequential (baseline)
   - Static Block
   - Static Cyclic
   - Static Block-Cyclic
   - Dynamic Task Queue
   - Dynamic Work-Stealing

3. **Parallelization Parameters**:
   - Thread counts from 2 to max hardware threads
   - Various chunk sizes to find optimal granularity

### Why These Tests

- **Different workloads**: To evaluate how each scheduler performs under various load distributions. For example, dynamic scheduling typically excels with imbalanced workloads.

- **Various chunk sizes**: Chunk size impacts load balancing and overhead. Too small causes excessive scheduling overhead, too large results in poor load balancing.

- **Thread scaling**: To measure performance scaling with increased thread count and identify potential bottlenecks.

### Measurement Methodology

- Each configuration is measured with multiple samples (10) and iterations per sample (20)
- Median execution time is used to reduce the impact of outliers
- Speedup is calculated relative to sequential execution for each specific workload

## Analyzing Results

### Generating Plots

To visualize the benchmark results:

```bash
cd scripts
python3 plot_benchmarks.py --all
```

or run for more info

```bash
python3 plot_benchmarks.py --help
```

This script reads the CSV results and generates various plots:

- Speedup vs. threads for each scheduler type
- Execution time vs. chunk size for different thread counts
- Comparative performance across different workloads

You will find the results in the `plots` directory.

### Key Metrics

The CSV file contains the following metrics:

- `WorkloadID`: Numerical ID of the workload
- `WorkloadDescription`: Description of the workload type
- `SchedulerName`: Name of the scheduling algorithm
- `SchedulerType`: General category (Sequential, Static, Dynamic)
- `StaticVariant`: For static schedulers, the specific approach (Block, Cyclic, Block-Cyclic)
- `DynamicVariant`: For dynamic schedulers, the specific approach (Task Queue, Work Stealing)
- `NumThreads`: Number of threads used
- `ChunkSize`: Size of work chunks (where applicable)
- `ExecutionTimeMs`: Measured execution time in milliseconds
- `BaselineTimeMs`: Sequential baseline time for the same workload
- `Speedup`: Ratio of baseline to execution time (higher is better)
