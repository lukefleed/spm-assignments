# Compiler and Standard
CXX := g++
CPP_STANDARD := -std=c++17

# Base Include Paths
INCLUDES := -Isrc/app -Isrc/core -Isrc/utils -Iminiz

# --- Compiler Flags ---
CXXFLAGS := -Wall -Wextra -pedantic $(CPP_STANDARD)
OPTFLAGS := -O3 -DNDEBUG
DBGFLAGS := -g -O0
OMPFLAGS := -fopenmp

# --- Linker Flags ---
LDFLAGS_BASE :=
LDFLAGS_OMP := -fopenmp

# --- Executable Names ---
TARGET_APP_PAR := minizp
TARGET_APP_SEQ := minizp_seq
TARGET_TEST := minizp_test
TARGET_BENCH := minizp_bench

# --- Source Files ---
# Core sources used by all targets (excluding main files)
CORE_SRCS := $(wildcard src/core/*.cpp) $(wildcard src/utils/*.cpp) miniz/miniz.c
# Main sources for each executable
APP_MAIN_SRC := src/app/minizp.cpp
TEST_MAIN_SRC := src/app/test_main.cpp
BENCH_MAIN_SRC := src/app/bench_main.cpp

# --- Object Directories ---
OBJDIR_PAR_RELEASE := obj/par/release
OBJDIR_SEQ_RELEASE := obj/seq/release

# --- Object File Generation ---
# Core Objects (Parallel Release Example)
CORE_OBJS_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(filter src/%.cpp, $(CORE_SRCS))) \
			   $(patsubst miniz/%.c,$(OBJDIR_PAR_RELEASE)/miniz.o, $(filter miniz/%.c, $(CORE_SRCS)))
# Main Object Files (Parallel Release Example)
APP_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(APP_MAIN_SRC))
TEST_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(TEST_MAIN_SRC))
BENCH_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(BENCH_MAIN_SRC))

# --- Core Object Compilation Rule (Parallel Release Example) ---
$(OBJDIR_PAR_RELEASE)/%.o: src/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling Core $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR_PAR_RELEASE)/miniz.o: miniz/miniz.c
	@mkdir -p $(@D)
	@echo "Compiling Miniz $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# --- Main Object Compilation Rule (Parallel Release Example) ---
$(OBJDIR_PAR_RELEASE)/app/%.o: src/app/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling Main $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# Add similar rules for Sequential and Debug builds...

# --- Linking Rules (Parallel Release Example) ---
$(TARGET_APP_PAR): $(APP_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (App Parallel Release)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

$(TARGET_TEST): $(TEST_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (Test Executable)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

$(TARGET_BENCH): $(BENCH_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (Benchmark Executable)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

# --- Build Targets ---
.PHONY: all app test bench clean cleanall help

help:
	@echo "MinizP: Parallel Compression/Decompression with OpenMP and Miniz"
	@echo ""
	@echo "MAKE TARGETS:"
	@echo "  all       Build parallel app, test, and bench executables"
	@echo "  app       Build minizp (parallel version, default)"
	@echo "  app_seq   Build minizp_seq (sequential-only version)"
	@echo "  test      Build and run correctness tests"
	@echo "  bench     Build and run all benchmarks"
	@echo "  clean     Remove executables and object files"
	@echo "  cleanall  Remove everything including test/benchmark data"
	@echo "  help      Display this help message"
	@echo ""
	@echo "CUSTOM MAKE VARIABLES:"
	@echo "  THREADS=N Set custom thread count for benchmarks (default: all cores)"
	@echo "            Example: make bench THREADS=4"
	@echo ""
	@echo "EXECUTABLES USAGE:"
	@echo ""
	@echo "minizp - Parallel compression/decompression tool"
	@echo "  Usage: ./minizp [OPTIONS] <file-or-directory> [more files or dirs]"
	@echo "  Options:"
	@echo "    -C <0|1>  Compress mode. 0=keep originals (default), 1=remove originals"
	@echo "    -D <0|1>  Decompress mode. 0=keep .zip files (default), 1=remove .zip files"
	@echo "    -r <0|1>  Recursive directory traversal. 0=disabled (default), 1=enabled"
	@echo "    -t <N>    Number of OpenMP threads (default: all cores)"
	@echo "    -q <L>    Verbosity level. 0=silent, 1=errors only (default), 2=verbose"
	@echo "    -h        Display help and exit"
	@echo "  Examples:"
	@echo "    ./minizp -C 0 -r 1 -t 8 -q 2 /path/to/dir"
	@echo "    ./minizp -D 1 file.zip"
	@echo ""
	@echo "minizp_seq - Sequential-only version (same options as minizp, no threading)"
	@echo ""
	@echo "minizp_test - Correctness tests"
	@echo "  Usage: ./minizp_test --mode=<seq|par>"
	@echo "  Examples:"
	@echo "    ./minizp_test --mode=seq"
	@echo "    ./minizp_test --mode=par"
	@echo ""
	@echo "minizp_bench - Benchmark driver"
	@echo "  Usage: ./minizp_bench --type=<TYPE> [OPTIONS]"
	@echo "  Required:"
	@echo "    --type=<TYPE>            Benchmark type:"
	@echo "                             one_large, many_small, many_large_sequential,"
	@echo "                             many_large_parallel, many_large_parallel_right"
	@echo "  Global Options (all types):"
	@echo "    --threads=<N>            Max threads to sweep (default: all cores)"
	@echo "    --iterations=<I>         Measurement iterations per config (default: 2)"
	@echo "    --warmup=<W>            Warmup runs before measurement (default: 1)"
	@echo "    --threshold=<bytes>      Large-file threshold (default: 16 MiB)"
	@echo "  Type-Specific Options:"
	@echo "    For --type=one_large:"
	@echo "      --large_size=<bytes>   Size for single large file (default: 512 MiB)"
	@echo "      --block_sizes_list=<S1,S2,...> Custom block sizes (default: 1-12 MiB)"
	@echo "    For --type=many_small:"
	@echo "      --num_small=<N>        Number of small files (default: 1000)"
	@echo "      --min_size=<bytes>     Min size for small files (default: 1 KiB)"
	@echo "      --max_size=<bytes>     Max size for small files (default: 1 MiB)"
	@echo "    For --type=many_large_* (sequential/parallel/parallel_right):"
	@echo "      --block_sizes_list=<S1,S2,...> Custom block sizes (default: 1-12 MiB)"
	@echo "      Note: Uses fixed 10 files of 50-250 MiB each"
	@echo "  Examples:"
	@echo "    ./minizp_bench --type=many_small --threads=8 --num_small=2000"
	@echo "    ./minizp_bench --type=one_large --threads=4 --large_size=1073741824"
	@echo "    ./minizp_bench --type=many_large_parallel --block_sizes_list=1048576,2097152"
	@echo ""
	@echo "plot.py - Generate plots from benchmark results"
	@echo "  Usage: ./plot.py [OPTIONS]"
	@echo "  Options:"
	@echo "    --one_large                Generate plots for one_large benchmarks"
	@echo "    --many_small               Generate plots for many_small"
	@echo "    --many_large_sequential    Generate plots for many_large_sequential"
	@echo "    --many_large_parallel      Generate plots for many_large_parallel"
	@echo "    --many_large_parallel_right Generate plots for many_large_parallel_right"
	@echo "    --all                      Generate all available plots"
	@echo "  Examples:"
	@echo "    ./plot.py --all"
	@echo "    ./plot.py --one_large --many_small"

all: $(TARGET_APP_PAR) $(TARGET_TEST) $(TARGET_BENCH) # Build app, test, and bench executables

app: $(TARGET_APP_PAR) # Build parallel release by default
app_seq: $(TARGET_APP_SEQ) # Specific target for sequential app

# --- Execution Targets ---
test: $(TARGET_TEST) # Depends directly on the test executable
	@echo "Running Correctness Tests (Sequential Mode)..."
	@./$(TARGET_TEST) --mode=seq
	@echo ""
	@echo "Running Correctness Tests (Parallel Mode)..."
	@./$(TARGET_TEST) --mode=par
	@echo ""
	@echo "Correctness Tests Finished."

bench: $(TARGET_BENCH) # Depends directly on the bench executable
	@echo "Running Benchmarks (Many Small Files)..."
	./$(TARGET_BENCH) --type=many_small \
			--threads=$(or $(THREADS),$(shell nproc)) \
			--iterations=2 \
			--warmup=0
	@echo ""
	@echo "Running Benchmarks (One Large File - Matrix)..."
	./$(TARGET_BENCH) --type=one_large \
			--threads=$(or $(THREADS),$(shell nproc)) \
			--iterations=2 \
			--warmup=0
	@echo ""
	@echo "Running Benchmarks (Many Large Files - Sequential Dispatch - Matrix)..."
	./$(TARGET_BENCH) --type=many_large_sequential \
			--threads=$(or $(THREADS),$(shell nproc)) \
			--iterations=2 \
			--warmup=0
	@echo ""
	@echo "Running Benchmarks (Many Large Files - Oversubscribed Parallel - Matrix)..."
	./$(TARGET_BENCH) --type=many_large_parallel \
			--threads=$(or $(THREADS),$(shell nproc)) \
			--iterations=2 \
			--warmup=0
	@echo ""
	@echo "Running Benchmarks (Many Large Files - Controlled Parallel - Matrix)..."
	./$(TARGET_BENCH) --type=many_large_parallel_right \
			--threads=$(or $(THREADS),$(shell nproc)) \
			--iterations=2 \
			--warmup=0
	@echo ""
	@echo "Benchmarks Finished."

# --- Clean Targets ---
clean:
	@echo "Cleaning object files and executables..."
	rm -f $(TARGET_APP_PAR) $(TARGET_APP_SEQ) $(TARGET_TEST) $(TARGET_BENCH) # Add debug targets if they exist
	rm -rf obj/

cleanall: clean
	@echo "Cleaning generated test/bench data..."
	rm -rf test_data_correctness_cpp/ test_data_bench_cpp/
