# CXX and MPI Compilers
CXX = g++
MPICXX = mpic++

# Compiler and Linker Flags
# Added -pthread for parallel framework support
# Common flags for all compilations
COMMON_FLAGS = -std=c++17 -Wall -Wextra -O3 -I./include -I./fastflow -pthread
# Specific CXXFLAGS for non-MPI compilation
CXXFLAGS = $(COMMON_FLAGS)
# Specific MPICXXFLAGS for MPI compilation
MPICXXFLAGS = $(COMMON_FLAGS)

# Linker flags - ensure pthreads is linked for parallel framework
LDFLAGS = -pthread

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# Common source files (adjust as needed)
COMMON_SRC = $(wildcard $(SRC_DIR)/common/*.cpp)
COMMON_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRC))

# Sequential Mergesort sources and objects
SEQ_SRC = $(wildcard $(SRC_DIR)/sequential/*.cpp)
SEQ_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SEQ_SRC))

# Hybrid Mergesort sources and objects
HYBRID_SRC = $(wildcard $(SRC_DIR)/hybrid/*.cpp)
HYBRID_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_SRC))

# Parallel Mergesort sources and objects
FF_SRC = $(wildcard $(SRC_DIR)/fastflow/*.cpp)
FF_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_SRC))

# Main application sources and objects (if any, e.g., main.cpp)
MAIN_SRC = $(wildcard $(SRC_DIR)/main/*.cpp)
MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_SRC))

# Test sources
TEST_SEQ_SRC = $(TEST_DIR)/test_sequential.cpp
TEST_PERF_SRC = $(TEST_DIR)/test_performance.cpp
TEST_CORRECT_SRC = $(TEST_DIR)/test_correctness.cpp
TEST_HYBRID_CORRECT_SRC = $(TEST_DIR)/test_hybrid_correctness.cpp
TEST_HYBRID_PERF_SRC = $(TEST_DIR)/test_hybrid_performance.cpp

# Test executables
TARGET_SEQ = $(BIN_DIR)/test_sequential
TARGET_PERF = $(BIN_DIR)/test_performance
TARGET_CORRECT = $(BIN_DIR)/test_correctness
TARGET_HYBRID_CORRECT = $(BIN_DIR)/test_hybrid_correctness
TARGET_HYBRID_PERF = $(BIN_DIR)/test_hybrid_performance

# Main executables
TARGET_SINGLE_NODE = $(BIN_DIR)/single_node_main
TARGET_MULTI_NODE = $(BIN_DIR)/multi_node_main

# Default target
all: $(TARGET_SEQ) $(TARGET_PERF) $(TARGET_CORRECT) $(TARGET_HYBRID_CORRECT) $(TARGET_HYBRID_PERF) $(TARGET_SINGLE_NODE) $(TARGET_MULTI_NODE)

# Rule to create object files for hybrid MPI sources
$(OBJ_DIR)/hybrid/%.o: $(SRC_DIR)/hybrid/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# Rule to create object files for main sources that use MPI
$(OBJ_DIR)/main/multi_node_main.o: $(SRC_DIR)/main/multi_node_main.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(MPICXX) $(MPICXXFLAGS) -c $< -o $@

# Rule to create object files (general case)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@) # Ensure the target directory for the object file exists
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to create object files from test directory (if any are compiled directly, usually not needed)
# $(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

# Ensure subdirectories for objects are also created
$(OBJ_DIR)/common $(OBJ_DIR)/sequential $(OBJ_DIR)/hybrid $(OBJ_DIR)/main $(OBJ_DIR)/fastflow:
	@mkdir -p $@

# Link test_sequential
$(TARGET_SEQ): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(SEQ_OBJ) $(TEST_SEQ_SRC)
	$(CXX) $(CXXFLAGS) $(TEST_SEQ_SRC) $(COMMON_OBJ) $(SEQ_OBJ) -o $@ $(LDFLAGS)

# Link test_performance (assuming it might use MPI, adjust if not)
# Added $(FF_OBJ) to the dependencies and linking command
$(TARGET_PERF): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(SEQ_OBJ) $(FF_OBJ) $(TEST_PERF_SRC)
	$(MPICXX) $(MPICXXFLAGS) $(TEST_PERF_SRC) $(COMMON_OBJ) $(SEQ_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Link test_correctness (assuming it might use MPI)
# Added $(FF_OBJ) to the dependencies and linking command
$(TARGET_CORRECT): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(SEQ_OBJ) $(HYBRID_OBJ) $(FF_OBJ) $(TEST_CORRECT_SRC)
	$(MPICXX) $(MPICXXFLAGS) $(TEST_CORRECT_SRC) $(COMMON_OBJ) $(SEQ_OBJ) $(HYBRID_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Link test_hybrid_correctness
# Added $(FF_OBJ) to the dependencies and linking command (HYBRID_OBJ likely includes ff_mergesort from its own cpp, but good to be explicit if it might be separate)
$(TARGET_HYBRID_CORRECT): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(HYBRID_OBJ) $(FF_OBJ) $(TEST_HYBRID_CORRECT_SRC)
	$(MPICXX) $(MPICXXFLAGS) $(TEST_HYBRID_CORRECT_SRC) $(COMMON_OBJ) $(HYBRID_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Link test_hybrid_performance
# Added $(FF_OBJ) to the dependencies and linking command
$(TARGET_HYBRID_PERF): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(HYBRID_OBJ) $(FF_OBJ) $(TEST_HYBRID_PERF_SRC)
	$(MPICXX) $(MPICXXFLAGS) $(TEST_HYBRID_PERF_SRC) $(COMMON_OBJ) $(HYBRID_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Link single_node_main executable
$(TARGET_SINGLE_NODE): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(SEQ_OBJ) $(FF_OBJ) $(OBJ_DIR)/main/single_node_main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/main/single_node_main.o $(COMMON_OBJ) $(SEQ_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Link multi_node_main executable
$(TARGET_MULTI_NODE): $(OBJ_DIR) $(BIN_DIR) $(COMMON_OBJ) $(SEQ_OBJ) $(HYBRID_OBJ) $(FF_OBJ) $(OBJ_DIR)/main/multi_node_main.o
	$(MPICXX) $(MPICXXFLAGS) $(OBJ_DIR)/main/multi_node_main.o $(COMMON_OBJ) $(SEQ_OBJ) $(HYBRID_OBJ) $(FF_OBJ) -o $@ $(LDFLAGS)

# Clean rule
clean:
	@echo "Cleaning up object and binary files..."
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@rm -f .baseline_time_hybrid_*.tmp

# Phony targets
.PHONY: all clean test_perf_hybrid test_correctness_single_node test_perf_single_node test_correctness_hybrid help

# Single node correctness test
test_correctness_single_node: $(TARGET_CORRECT)
	@echo "Running single node correctness tests..."
	$(TARGET_CORRECT)

# Default values for single node performance test
# Generate thread list dynamically: 2, 4, 6, 8, ..., up to max system threads
MAX_THREADS := $(shell nproc)
THREAD_LIST ?= $(shell seq 2 2 $(MAX_THREADS))
ARRAY_SIZE_M ?= 10
PAYLOAD_SIZE_B ?= 64

# Configurable single node performance test
test_perf_single_node: $(TARGET_PERF)
	@echo "=============================================================================="
	@echo "                   Single Node Performance Analysis                           "
	@echo "------------------------------------------------------------------------------"
	@echo " Config: $(ARRAY_SIZE_M)M records, $(PAYLOAD_SIZE_B)B payload"
	@echo " Threads: $(THREAD_LIST)"
	@echo " Optional tests: --size-scaling --payload-scaling"
	@echo "=============================================================================="
	$(TARGET_PERF) "$(THREAD_LIST)" $(ARRAY_SIZE_M) $(PAYLOAD_SIZE_B) $(EXTRA_FLAGS)
	@echo "------------------------------------------------------------------------------"
	@echo " Analysis Complete. Results saved to: performance_results.csv"
	@echo "=============================================================================="

# Hybrid correctness test
test_correctness_hybrid: $(TARGET_HYBRID_CORRECT)
	@echo "Running hybrid MPI+parallel correctness tests..."
	mpirun --oversubscribe -np 2 $(TARGET_HYBRID_CORRECT)

# Help target with enhanced information
help:
	@echo "Hybrid MPI+Parallel Mergesort Build System"
	@echo "==========================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all                          - Build all test executables and main programs"
	@echo "  clean                        - Remove all build artifacts"
	@echo ""
	@echo "Main Programs:"
	@echo "  $(TARGET_SINGLE_NODE)        - Single-node parallel mergesort application"
	@echo "  $(TARGET_MULTI_NODE)         - Multi-node hybrid MPI+parallel mergesort application"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test_correctness_single_node - Verify single-node implementation correctness"
	@echo "  test_correctness_hybrid      - Verify hybrid MPI+parallel correctness"
	@echo "  test_perf_single_node        - Single-node performance analysis"
	@echo "  test_perf_hybrid             - Dual-baseline hybrid performance analysis"
	@echo ""
	@echo "Hybrid Performance Test Configuration:"
	@echo "  MPI_NODES_LIST=\"1 2 4 8\"     - MPI process counts to test"
	@echo "  FF_THREADS=4                 - FastFlow threads per MPI process"
	@echo "  RECORDS_SIZE_M=10            - Dataset size in millions of records"
	@echo "  PAYLOAD_SIZE_B=64            - Record payload size in bytes"
	@echo "  HYBRID_CSV_FILE=results.csv  - Output CSV file for detailed analysis"
	@echo ""
	@echo "Example Usage:"
	@echo "  make test_perf_hybrid MPI_NODES_LIST=\"1 2 4\" FF_THREADS=8"
	@echo "  make test_perf_hybrid RECORDS_SIZE_M=50 PAYLOAD_SIZE_B=128"

# Hybrid MPI+Parallel performance test with dual baseline analysis
MPI_NODES_LIST ?= 1 2 4 8
FF_THREADS ?= 4
RECORDS_SIZE_M ?= 10
PAYLOAD_SIZE_B ?= 64
HYBRID_CSV_FILE ?= hybrid_performance_results.csv

# Hybrid MPI+Parallel performance test with dual baseline analysis
test_perf_hybrid: $(TARGET_HYBRID_PERF)
	@echo "=============================================================================="
	@echo "                 Hybrid MPI+Parallel Performance Test                        "
	@echo "------------------------------------------------------------------------------"
	@echo " Config: $(RECORDS_SIZE_M)M records, $(PAYLOAD_SIZE_B)B payload, $(FF_THREADS) FF threads/process"
	@echo " MPI Processes: $(MPI_NODES_LIST)"
	@echo " Analysis: Parallel baseline comparison with MPI scaling"
	@echo " Output: $(HYBRID_CSV_FILE)"
	@echo "=============================================================================="
	@# Run baseline establishment first (single-node parallel)
	@baseline_time=$$(mpirun --oversubscribe -np 1 $(TARGET_HYBRID_PERF) $(FF_THREADS) $(RECORDS_SIZE_M) $(PAYLOAD_SIZE_B) $(HYBRID_CSV_FILE) --quiet 2>&1 | grep "Mergesort FF" | awk '{print $$3}'); \
	echo ""; \
	echo "MPI Procs       Time (ms)Throughput (MRec/s) 	    Speedup    MPI Eff (%)  Total Eff (%)"; \
	echo "--------------------------------------------------------------------------------------------"; \
	for nodes in $(MPI_NODES_LIST); do \
		if [ $$nodes -eq 1 ]; then \
			mpirun --oversubscribe -np $$nodes $(TARGET_HYBRID_PERF) $(FF_THREADS) $(RECORDS_SIZE_M) $(PAYLOAD_SIZE_B) $(HYBRID_CSV_FILE) --quiet; \
		else \
			mpirun --oversubscribe -np $$nodes $(TARGET_HYBRID_PERF) $(FF_THREADS) $(RECORDS_SIZE_M) $(PAYLOAD_SIZE_B) $(HYBRID_CSV_FILE) --quiet --skip-baselines --baseline-time=$$baseline_time; \
		fi; \
	done
	@echo "=============================================================================="

# Performance analysis targets for different scenarios
.PHONY: test_perf_hybrid_scaling test_perf_hybrid_threads

# MPI scaling test with fixed thread count
test_perf_hybrid_scaling: $(TARGET_HYBRID_PERF)
	@echo "=== MPI Scaling Analysis (Fixed $(FF_THREADS) threads/process) ==="
	@make test_perf_hybrid MPI_NODES_LIST="1 2 4 8 16" FF_THREADS=$(FF_THREADS)

# Thread scaling test with single MPI process
test_perf_hybrid_threads: $(TARGET_HYBRID_PERF)
	@echo "=== Thread Scaling Analysis (Single MPI process) ==="
	@for threads in 1 2 4 8 12 16; do \
		echo "Testing $$threads threads..."; \
		mpirun --oversubscribe -np 1 $(TARGET_HYBRID_PERF) $$threads $(RECORDS_SIZE_M) $(PAYLOAD_SIZE_B) thread_scaling_$(HYBRID_CSV_FILE) --quiet; \
	done
