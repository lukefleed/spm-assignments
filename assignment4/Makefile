# Compiler and flags
CXX = g++
MPICXX = mpicxx
CXX_STD = -std=c++17
# -O3 enables a high level of optimization.
# -march=native tailors optimizations to the specific architecture of the compilation machine.
OPTIM_FLAGS = -O3 -march=native
# -g adds debugging information.
# -Wall, -Wextra, -Wpedantic enable a comprehensive set of warnings for robust code.
DEBUG_FLAGS = -g -Wall -Wextra -Wpedantic
COMMON_FLAGS = $(CXX_STD) $(DEBUG_FLAGS)

# FastFlow specific paths: FF_HOME should be set to your FastFlow installation directory.
# Example: export FF_HOME=/opt/fastflow
# This Makefile requires FF_HOME to be set in the environment.
ifndef FF_HOME
    $(error FF_HOME is not set. Please set it to your FastFlow installation path, e.g., export FF_HOME=/path/to/fastflow)
endif
FF_INC = -I$(FF_HOME)/include
# Common FastFlow libraries; adjust if your setup requires more specific linking.
FF_LIB = -L$(FF_HOME)/lib -lff

# MPI flags are typically managed by the mpicxx wrapper.
MPI_INC =
MPI_LIB =

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
SCRIPT_DIR = scripts
DATA_DIR = data

# Common object files used by both executables
COMMON_SRCS = $(SRC_DIR)/utils.cpp $(SRC_DIR)/performance_timer.cpp $(SRC_DIR)/mergesort_common.cpp
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))

# Single-node (FastFlow) specific files and executable
FF_MAIN_SRC = $(SRC_DIR)/main_ff.cpp
FF_MODULE_SRCS = $(SRC_DIR)/mergesort_ff.cpp
FF_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_MODULE_SRCS))
FF_MAIN_OBJ = $(OBJ_DIR)/main_ff.o
EXECUTABLE_FF = $(BIN_DIR)/sort_ff

# Hybrid (MPI + FastFlow) specific files and executable
HYBRID_MAIN_SRC = $(SRC_DIR)/main_hybrid.cpp
# Hybrid version reuses the FastFlow module for intra-node sorting.
HYBRID_MODULE_SRCS = $(SRC_DIR)/mergesort_hybrid.cpp $(SRC_DIR)/mergesort_ff.cpp
HYBRID_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_MODULE_SRCS))
HYBRID_MAIN_OBJ = $(OBJ_DIR)/main_hybrid.o
EXECUTABLE_HYBRID = $(BIN_DIR)/sort_hybrid

# Default target: build all executables.
.PHONY: all clean directories test_correctness_single_node test_correctness_hybrid run_perf_single_node run_perf_hybrid run_all_tests plot_results data

all: directories $(EXECUTABLE_FF) $(EXECUTABLE_HYBRID)

directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(DATA_DIR) $(BUILD_DIR)/plots

# Generic rule for compiling common C++ source files.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for compiling the FastFlow main source file.
$(FF_MAIN_OBJ): $(FF_MAIN_SRC) $(INC_DIR)/record.h $(INC_DIR)/utils.h $(INC_DIR)/mergesort_ff.h $(INC_DIR)/performance_timer.h
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

# Rule for compiling FastFlow specific modules.
$(OBJ_DIR)/mergesort_ff.o: $(SRC_DIR)/mergesort_ff.cpp $(INC_DIR)/mergesort_ff.h $(INC_DIR)/record.h $(INC_DIR)/mergesort_common.h
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

# Rule for compiling the Hybrid main source file.
$(HYBRID_MAIN_OBJ): $(HYBRID_MAIN_SRC) $(INC_DIR)/record.h $(INC_DIR)/utils.h $(INC_DIR)/mergesort_hybrid.h $(INC_DIR)/performance_timer.h
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@

# Rule for compiling Hybrid specific modules.
$(OBJ_DIR)/mergesort_hybrid.o: $(SRC_DIR)/mergesort_hybrid.cpp $(INC_DIR)/mergesort_hybrid.h $(INC_DIR)/record.h $(INC_DIR)/mergesort_ff.h
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@

# Linking single-node (FastFlow) executable.
# pthread is commonly required by FastFlow.
$(EXECUTABLE_FF): $(COMMON_OBJS) $(FF_MODULE_OBJS) $(FF_MAIN_OBJ)
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ $(FF_LIB) -lpthread -o $@

# Linking hybrid (MPI + FastFlow) executable.
$(EXECUTABLE_HYBRID): $(COMMON_OBJS) $(HYBRID_MODULE_OBJS) $(HYBRID_MAIN_OBJ)
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ $(FF_LIB) $(MPI_LIB) -lpthread -o $@

clean:
	rm -rf $(BUILD_DIR)/*
	rm -f results_*.csv # Clean up result files from benchmark runs.
	rm -f $(DATA_DIR)/input_N*_R*.dat # Clean up generated data files

# --- Test Targets ---
# Default parameters for test runs. These can be overridden from the command line.
# Example: make test_correctness_single_node N=10000 R_PAYLOAD=8
N ?= 1000000
R_PAYLOAD ?= 64
T_THREADS ?= 4
P_PROCS ?= 2 # Number of MPI processes

# Base name for input files, constructed from N and R_PAYLOAD.
INPUT_FILE_BASE = $(DATA_DIR)/input_N$(N)_R$(R_PAYLOAD)
FF_ARGS = -s $(N) -r $(R_PAYLOAD) -t $(T_THREADS)
HYBRID_ARGS = -s $(N) -r $(R_PAYLOAD) -t $(T_THREADS) # -p P is implicit via mpirun

# Target to generate test data using the python script.
# This is a prerequisite for tests requiring specific data files.
# Define data file generation as a pattern rule for clarity.
$(DATA_DIR)/input_N%_R%.dat: $(SCRIPT_DIR)/generate_data.py
	@echo "Generating test data: $@ (N=$*, R=$**)"
	python $(SCRIPT_DIR)/generate_data.py --size $(*) --payload $(**) --output $@

# Make data files explicit prerequisites where used.
# Example: if N=1000, R_PAYLOAD=64, it expects $(DATA_DIR)/input_N1000_R64.dat
TARGET_DATA_FILE = $(INPUT_FILE_BASE).dat

test_correctness_single_node: $(EXECUTABLE_FF) $(TARGET_DATA_FILE)
	@echo "Running Correctness Test for Single-Node: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS)"
	./$(EXECUTABLE_FF) $(FF_ARGS) --input $(TARGET_DATA_FILE) --check_correctness

test_correctness_hybrid: $(EXECUTABLE_HYBRID) $(TARGET_DATA_FILE)
	@echo "Running Correctness Test for Hybrid: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS), P=$(P_PROCS)"
	mpirun -np $(P_PROCS) --oversubscribe ./$(EXECUTABLE_HYBRID) $(HYBRID_ARGS) --input $(TARGET_DATA_FILE) --check_correctness

# --- Performance Benchmark Suites ---
# These targets invoke the benchmark script for comprehensive performance analysis.
# Define default value lists for iterating through parameters.
N_VALUES_LIST ?= "10000000 100000000" # Example: 10M, 100M records
R_VALUES_LIST ?= "8 64 256"           # Example: 8, 64, 256 byte payloads
T_VALUES_LIST ?= "1 2 4 8 16 32"      # For FastFlow threads
P_VALUES_LIST ?= "1 2 4 8"            # For MPI processes/nodes (up to 8 as per assignment)

run_perf_single_node: $(EXECUTABLE_FF) $(SCRIPT_DIR)/run_benchmarks.sh
	@echo "Starting Single-Node Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh single_node "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)"

run_perf_hybrid: $(EXECUTABLE_HYBRID) $(SCRIPT_DIR)/run_benchmarks.sh
	@echo "Starting Hybrid Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh hybrid "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)" "$(P_VALUES_LIST)"

# A target to run a minimal set of tests for quick verification.
run_all_tests:
	$(MAKE) test_correctness_single_node N=10000 R_PAYLOAD=8 T_THREADS=2
	$(MAKE) test_correctness_hybrid N=10000 R_PAYLOAD=8 T_THREADS=2 P_PROCS=2
	# Run a very small performance benchmark for CI-like checks
	$(MAKE) run_perf_single_node N_VALUES_LIST="100000" R_VALUES_LIST="64" T_VALUES_LIST="1 2"
	$(MAKE) run_perf_hybrid N_VALUES_LIST="100000" R_VALUES_LIST="64" T_VALUES_LIST="1" P_VALUES_LIST="1 2"

plot_results: $(SCRIPT_DIR)/plot_results.py
	@echo "Generating plots from benchmark results..."
	# Ensure result files exist before trying to plot, or handle missing files in script.
	if [ -f results_single_node.csv ] && [ -f results_hybrid.csv ]; then \
		python $(SCRIPT_DIR)/plot_results.py \
			--input_ff results_single_node.csv \
			--input_hybrid results_hybrid.csv \
			--output_dir $(BUILD_DIR)/plots; \
	else \
		echo "Warning: One or both result CSV files not found. Skipping plot generation."; \
	fi

# The 'data' target is a phony target to group data generation.
# Individual data files are generated as needed by their explicit rules.
data:
	@echo "Data generation is handled on-demand by specific test targets."
	@echo "Example: 'make test_correctness_single_node N=10000 R_PAYLOAD=32' will create data/input_N10000_R32.dat if needed."
