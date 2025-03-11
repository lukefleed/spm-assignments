# Softmax Benchmarking

This repository contains implementations and benchmarking tools for the softmax function using different optimization techniques. The project explores plain, auto-vectorized, and manually AVX-vectorized implementations with analysis of performance, numerical stability, and thread scaling.

## Project Structure

```
assignment1/
├── include/               # Header files including AVX math functions
├── report/                # LaTeX report
├── images/                # Generated performance plots
├── results/               # Benchmark results in CSV format
├── softmax_plain.cpp      # Plain implementation
├── softmax_auto.cpp       # Auto-vectorized implementation
├── softmax_avx.cpp        # AVX-vectorized implementation
├── softmax_test.cpp       # Benchmark and testing framework
├── plot.py                # Script for generating performance plots
├── plot_scaling.py        # Script for thread scaling plots
└── Makefile               # Build and test automation
```

## Building the Project

### Basic Build Commands

```bash
# Build all implementations
make all

# Build and run the test suite (uses AVX512 and parallelization)
make test

# Clean build artifacts
make clean

# Clean all generated files
make cleanall
```

## Testing Commands

### Performance Testing

```bash
# Run all performance tests
make test-performance

# Run specific configurations:
make test-parallel-avx512      # Parallel execution with AVX512
make test-parallel-noavx512    # Parallel execution without AVX512
make test-noparallel-avx512    # Sequential execution with AVX512
make test-noparallel-noavx512  # Sequential execution without AVX512
```

### Stability Testing

```bash
# Run numerical stability test
make test-stability
```

### Thread Scaling Testing

```bash
# Run thread scaling test (always uses AVX512)
make test-scaling
```

### Run All Tests

```bash
# Run performance, stability, and thread scaling tests
make test-all
```

## Data Visualization

### Performance and Speedup Plots

Use plot.py to generate performance comparison charts:

```bash
# Generate plots for parallel execution with AVX512
python3 plot.py --parallel --avx512

# Generate plots for parallel execution without AVX512
python3 plot.py --parallel --noavx512

# Generate plots for sequential execution with AVX512
python3 plot.py --noparallel --avx512

# Generate plots for sequential execution without AVX512
python3 plot.py --noparallel --noavx512

# Add numerical stability analysis to any of the above commands
python3 plot.py --parallel --avx512 --stability
```

### Thread Scaling Plots

```bash
# Generate thread scaling plots
python3 plot_scaling.py
```

## Output

- Performance results are saved to `results/results_[config].csv`
- Speedup data is saved to `results/speedup_[config].csv`
- Stability data is saved to `results/stability_[config].csv`
- Thread scaling data is saved to `results/thread_scaling/thread_scaling_avx512.csv`

Generated plots are saved in the `images/` directory as PDF files:

- Performance: `images/performance/softmax_[config].pdf`
- Small sizes: `images/performance/softmax_[config]_small.pdf`
- Speedup: `images/speedup/softmax_speedup_[config].pdf`
- Stability: `images/stability/stability_[config].pdf`
- Thread scaling: `images/thread_scaling/thread_scaling/thread_scaling_avx512.pdf`

Where `[config]` represents the configuration (e.g., `parallel_avx512`).

# Reproducibility Guidelines

## Experimental Configuration

The default benchmark configuration is designed for comprehensive analysis with extensive iterations and large datasets, which may require significant computational time. For preliminary evaluations or systems with limited resources, we recommend modifying the following parameters in `softmax_test.cpp`:

1. Reduce the number of `samples` (default: 15)
2. Decrease `iterations_per_sample` (default: 30)
3. Lower the maximum dataset size (default: 2^22 elements)

These adjustments will substantially reduce execution time while still providing meaningful initial results.

## Execution Protocol

To execute the complete test suite with modified parameters:

```bash
make test-all
```

## Data Visualization

Generate the comprehensive set of performance visualizations using:

```bash
python3 plot.py --all --stability
python3 plot_scaling.py
```

## Hardware Considerations

The benchmark framework distinguishes between AVX512-enabled and non-AVX512 configurations. It is important to note that on processors lacking AVX512 support, the `softmax_auto` implementation's performance metrics will remain invariant between AVX512 and non-AVX512 compilation flags. The `softmax_avx` implementation employs AVX2 instruction sets regardless of the AVX512 compilation setting.
