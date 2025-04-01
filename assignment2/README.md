This project implements and benchmarks various scheduling algorithms for calculating the maximum number of steps in the Collatz sequence (3n+1 problem) across ranges of numbers. It provides a comprehensive framework for comparing different parallel execution strategies and analyzing their performance.

## Building the Project

To build the project, simply run:

```bash
make
```

This will compile all source files and create the `collatz_par` executable in the build directory.

For a clean build:

```bash
make clean && make
```

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

Results are saved to `results/performance_results.csv`.

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

### Measurement Methodology

- Each configuration is measured with multiple samples (10) and iterations per sample (50)
- Median execution time is used to reduce the impact of outliers
- Speedup is calculated relative to sequential execution for each specific workload

## Analyzing Results

### Generating Plots

To visualize the benchmark results:

```bash
cd scripts
python3 plot_benchmarks.py --all
```

For more information and options:

```bash
python3 plot_benchmarks.py --help
```

This script reads the CSV results and generates various plots:

- Speedup vs. threads for each scheduler type
- Execution time vs. chunk size for different thread counts
- Comparative performance across different workloads
- Heatmaps showing the relationship between chunk size, thread count, and speedup

Generated plots are saved in the `results/plots/` directory with subdirectories for each visualization type.
