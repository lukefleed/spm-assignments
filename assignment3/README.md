# MinizP: Parallel Compression/Decompression with OpenMP and Miniz

This project implements a parallel compressor/decompressor based on the Miniz library and OpenMP for DEFLATE compression of both small and large files (using block-based parallelism for large inputs).

## Project Structure

```text
assignment3/
├── Makefile                    # Build script (parallel, seq, test, bench targets)
├── README.md                   # Main documentation (this file)
├── plot.py                     # Python script to generate plots from CSV results
├── miniz/                      # Miniz library source (miniz.c + miniz.h)
├── src/
│   ├── app/                    # Executables and CLI parsing
│   │   ├── minizp.cpp          # Main program (CLI for compress/decompress)
│   │   ├── test_main.cpp       # Unit tests driver (seq and par modes)
│   │   ├── bench_main.cpp      # Benchmark driver (thread/block sweeps)
│   │   ├── cmdline.hpp         # Command-line options parser
│   │   └── config.hpp          # Configuration data and default constants
│   ├── core/                   # Core logic: file discovery, compress/decompress
│   │   ├── file_handler.hpp    # File discovery and filtering
│   │   ├── file_handler.cpp
│   │   ├── compressor.hpp      # Interface for small/large compression
│   │   └── compressor.cpp      # Implementation using Miniz, mmap, OpenMP
│   └── utils/                  # Utilities for testing and benchmarking
│       ├── test_utils.hpp      # Random file creation, file comparison, cleanup
│       ├── test_utils.cpp
│       ├── bench_utils.hpp     # Timer and benchmark runner
│       └── bench_utils.cpp
└── obj/                        # (generated) object files and directories
```

### File Descriptions

- **Makefile**: Defines build targets: `all` (parallel app, tests, bench), `app_seq`, `test`, `bench`, `clean`, `cleanall`.
- **plot.py**: Generates three types of plots (heatmap, speedup vs threads, block-size speedup) from CSV benchmarks.
- **miniz/**: Single-file Miniz implementation of DEFLATE.
- **src/app/**
  - _minizp.cpp_: Parses CLI, discovers files via FileHandler, and runs compression/decompression in parallel.
  - _test_main.cpp_: Sets up test environment, runs correctness tests for both sequential and parallel modes.
  - _bench_main.cpp_: Sets up benchmark environment (one large or many small files), measures performance over various threads and block sizes, writes CSV.
  - _cmdline.hpp_: Defines CLI options `-C`, `-D`, `-r`, `-t`, `-q`, and help.
  - _config.hpp_: Defines `ConfigData` (mode, thresholds, block size, threads, verbosity).
- **src/core/**
  - _file_handler._: Recursively discovers and filters files based on mode and suffix.
  - _compressor._: Implements small-file compression with single-threaded zlib stream and large-file block-based parallel compression with custom header and metadata.
- **src/utils/**
  - _test_utils._: Helpers to generate random files, compare files byte-by-byte, and clean directories.
  - _bench_utils._: High-resolution timer and `run_benchmark` function to measure median execution time.

## Compression Strategy

1. **Small Files** (< threshold): Memory-map input (`mmap`), compress in a single thread using Miniz, write output `.zip` with a 64-bit original-size header.
2. **Large Files** (>= threshold): Memory-map input, split into blocks of size `block_size`, compress each block in parallel (OpenMP) into temporary buffers. After compression, memory-map the output file, write a custom header (`magic`, `version`, `original_size`, `num_blocks`), block-size metadata, and parallel-copy compressed blocks to reduce system call overhead.
3. **Decompression**: Mirror compression logic. For large format, read header and metadata, memory-map input and output, decompress blocks in parallel to the output map.
4. **File Discovery**: Traverse directories (optionally recursively), include or skip files based on `.zip` suffix and compression/decompression mode.

## Usage Instructions

### Prerequisites

- GNU Make, g++ with C++17 and OpenMP support
- Python3 with `pandas`, `plotly`, and `kaleido` for plotting: `pip install pandas plotly kaleido`

### Building and Testing

```bash
cd assignment3
make all             # Build parallel app, test, and bench executables
make app_seq         # Build sequential-only app
make test            # Run correctness tests (seq and par)
make bench           # Run benchmarks for many_small and one_large
make clean           # Remove executables and object files
make cleanall        # Also removes test and benchmark data directories
```

### Manual Execution of Executables

#### Parallel App (`minizp`)

```bash
./minizp [options] <file-or-dir> [more files or dirs]
```

Options:

- `-C 0|1`: compression mode; `0` preserves originals (default), `1` removes originals
- `-D 0|1`: decompression mode; `0` preserves `.zip`, `1` removes `.zip`
- `-r 0|1`: recursion into subdirectories; default `0`
- `-t N`: number of OpenMP threads; default is max available
- `-q L`: verbosity level (`0`=silent, `1`=errors, `2`=verbose); default `1`
- `-h`: display help

Examples:

```bash
# Compress directory recursively, keep originals, 8 threads, verbose
./minizp -C 0 -r 1 -t 8 -q 2 /path/to/dir
# Decompress and remove .zip files
./minizp -D 1 /path/to/file.zip
```

#### Sequential App (`minizp_seq`)

Same usage, but built without OpenMP or by specifying `-t 1`.

### Benchmark CSV Output

After running `make bench` or manually invoking `minizp_bench`, two CSV files are generated in the `assignment3/` directory:

- `benchmark_many_small.csv`: columns `threads,seq_time_s,par_time_s,speedup`
- `benchmark_matrix_results.csv`: columns `block_size,threads,seq_time_s,par_time_s,speedup`

### Generating Plots

Use `plot.py` to create PDF plots:

```bash
./plot.py --one_large       # Heatmap: speedup vs threads & block size
./plot.py --many_small      # Speedup vs threads for many small files with Amdahl curve
./plot.py --block_speedup   # Speedup vs threads for selected block sizes
./plot.py --all             # Generate all plots
```

Plots are saved under `results/plots/<type>/`, for example:

- `results/plots/one_large/speedup_matrix_one_large.pdf`
- `results/plots/many_small/speedup_many_small.pdf`
- `results/plots/block_speedup/block_<MiB>MiB_speedup.pdf`
