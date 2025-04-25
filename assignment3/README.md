# MinizP: Parallel Compression/Decompression with OpenMP and Miniz

This project implements a parallel compressor/decompressor based on the Miniz library and OpenMP for DEFLATE compression of both small and large files (using block-based parallelism for large inputs). It includes benchmarks for various parallelization strategies.

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
│   │   ├── bench_main.cpp      # Benchmark driver
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
├── obj/                        # (generated) object files and directories
└── results/                    # (generated) benchmark data and plots
    ├── data/                   # CSV benchmark results
    └── plots/                  # PDF plots
```

### File Descriptions

- **Makefile**: Defines build targets: `all` (parallel app, tests, bench), `app_seq`, `test`, `bench`, `clean`, `cleanall`.
- **plot.py**: Generates plots (heatmaps, speedup vs threads) from CSV benchmarks.
- **miniz/**: Single-file Miniz implementation of DEFLATE.
- **src/app/**
  - _minizp.cpp_: Parses CLI, discovers files via FileHandler, and runs compression/decompression in parallel.
  - _test_main.cpp_: Sets up test environment, runs correctness tests for both sequential and parallel modes.
  - _bench_main.cpp_: Sets up benchmark environment (one large, many small, or many large files), measures performance over various threads and block sizes for different parallel strategies (`one_large`, `many_small`, `many_large_sequential`, `many_large_parallel`, `many_large_parallel_right`), writes CSVs.
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
make bench           # Run all benchmarks (many_small, one_large, many_large_*)
make clean           # Remove executables and object files
make cleanall        # Also removes test and benchmark data directories
```

## CLI Reference

### minizp

**Synopsis**

```
./minizp [OPTIONS] <file-or-directory> [more files or dirs]
```

**Description**
Parallel compression/decompression tool using Miniz and OpenMP. Processes files or directories according to mode flags.

**Options**

- `-C <0|1>`
  - Set compress mode. `0` = compress and **keep** originals (default), `1` = compress and **remove** originals after `.zip` creation.
- `-D <0|1>`
  - Set decompress mode. `0` = decompress and **keep** `.zip` files (default), `1` = decompress and **remove** `.zip` files after extraction.
- `-r <0|1>`
  - Recursive directory traversal. `0` = disabled (default), `1` = enabled.
- `-t <N>`
  - Number of OpenMP threads for parallel regions. Must be a positive integer. Default = maximum available cores.
- `-q <L>`
  - Verbosity level. `0` = silent, `1` = errors only (default), `2` = verbose logging.
- `-h`
  - Display help/usage and exit.

**Examples**

- Compress a directory recursively with 8 threads in verbose mode, preserving originals:

  ```bash
  ./minizp -C 0 -r 1 -t 8 -q 2 /path/to/dir
  ```

- Decompress a single file and remove the `.zip` archive:

  ```bash
  ./minizp -D 1 /path/to/file.zip
  ```

**Description**
Sequential-only build (no OpenMP). Same flags as `minizp`, but single-threaded regardless of `-t`.

### minizp_bench

**Synopsis**

```
./minizp_bench
  --type=<TYPE>
  [--threads=<N>]
  [--iterations=<I>]
  [--warmup=<W>]
  [--large_size=<bytes>]
  [--num_small=<N>]
  [--min_size=<bytes>] [--max_size=<bytes>]
  [--verbosity=<L>]
  [--threshold=<bytes>]
  [--blocksize=<bytes>]
  [--block_sizes_list=<S1,S2,...>]
```

**Description**
Benchmark driver that generates test data, sweeps over thread counts and block sizes, measures sequential and parallel performance, and writes CSV results under `results/data/`.

**Options**

- `--type=<TYPE>`
  - Type of benchmark (required). One of:
    - `one_large`
    - `many_small`
    - `many_large_sequential`
    - `many_large_parallel`
    - `many_large_parallel_right`
- `--threads=<N>`
  - Maximum number of threads to sweep in outer or inner loops. Default = max available cores.
- `--iterations=<I>`
  - Number of timed measurement iterations per configuration. Default = 2.
- `--warmup=<W>`
  - Number of warmup runs before measurement. Default = 1.
- `--large_size=<bytes>`
  - Size for single large file in `one_large` type. Default = 512 MiB.
- `--num_small=<N>`
  - Number of small files to generate for `many_small`. Default = 4000.
- `--min_size=<bytes>` / `--max_size=<bytes>`
  - Size range for small-file generation. Defaults = 1 KiB / 1 MiB.
- `--verbosity=<L>`
  - Verbosity level (0-2). Default = 0.
- `--threshold=<bytes>`
  - Override large-file threshold (bytes). Default = 16 MiB.
- `--blocksize=<bytes>`
  - Block size for large-file sweeps. Default = 1 MiB.
- `--block_sizes_list=<S1,S2,...>`
  - Comma-separated list of block sizes (bytes) to override default matrix.

**Examples**

- Benchmark many small files up to 8 threads:

  ```bash
  ./minizp_bench --type=many_small --threads=8
  ```

- One large file matrix over custom block sizes:

  ```bash
  ./minizp_bench --type=one_large --threads=4 --block_sizes_list=1048576,2097152,4194304
  ```

### plot.py

**Synopsis**

```
./plot.py [OPTIONS]
```

**Description**
Generate PDF plots from benchmark CSV files located in `results/data/` and save under `results/plots/`.

**Options**

- `--one_large`
  - Generate plots for `one_large` benchmarks.
- `--many_small`
  - Generate plots for `many_small`.
- `--many_large_sequential`
  - Generate plots for `many_large_sequential`.
- `--many_large_parallel`
  - Generate plots for `many_large_parallel`.
- `--many_large_parallel_right`
  - Generate plots for `many_large_parallel_right`.
- `--all`
  - Generate all available plots.

**Examples**

- Generate all plots:

  ```bash
  ./plot.py --all
  ```

## Test Script Edge Cases

The provided `test_minizp.sh` script ensures the following scenarios are correctly handled:

- Empty directories (e.g., `nested/emptydir`)
- Symlinked directories (e.g., `nested_symlink`)
- Mixed input order (processing a directory and files in any order)
