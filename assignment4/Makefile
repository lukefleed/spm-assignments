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

# Rule to create object files
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

# Single node performance test
test_perf_single_node: $(TARGET_PERF)
	@echo "Running single node performance benchmarks..."
	@echo "This may take several minutes to complete..."
	$(TARGET_PERF)

# Hybrid correctness test
test_correctness_hybrid: $(TARGET_HYBRID_CORRECT)
	@echo "Running hybrid MPI+parallel correctness tests..."
	mpirun --oversubscribe -np 2 $(TARGET_HYBRID_CORRECT)

# Help target
help:
	@echo "Available targets:"
	@echo "  all                          - Build all test executables and main programs"
	@echo "  clean                        - Remove all build artifacts"
	@echo ""
	@echo "Main Programs:"
	@echo "  $(TARGET_SINGLE_NODE)        - Single-node parallel mergesort application"
	@echo "  $(TARGET_MULTI_NODE)         - Multi-node hybrid MPI+parallel mergesort application"
	@echo ""
	@echo "Single Node Tests (Parallel implementation):"
	@echo "  test_correctness_single_node - Run correctness tests for single node"
	@echo "  test_perf_single_node        - Run performance benchmarks for single node"
	@echo ""
	@echo "Hybrid Tests (MPI + Parallel):"
	@echo "  test_correctness_hybrid      - Run correctness tests for hybrid MPI+parallel"
	@echo "  test_perf_hybrid             - Run hybrid MPI+parallel performance analysis"
	@echo ""
	@echo "Customizable variables for test_perf_hybrid:"
	@echo "  MPI_NODES_LIST=\"1 2 4 8\"     - List of MPI process counts to test"
	@echo "  FF_THREADS=4                 - Number of parallel threads per process"
	@echo "  RECORDS_SIZE_M=100           - Number of records in millions"
	@echo "  PAYLOAD_SIZE_B=64            - Payload size in bytes"
	@echo ""
	@echo "Examples:"
	@echo "  make test_perf_hybrid MPI_NODES_LIST=\"1 2 3 4\" FF_THREADS=8"
	@echo "  make test_perf_hybrid FF_THREADS=2"
	@echo "  make help"

# Default values for MPI nodes and FF threads for test_perf_hybrid
MPI_NODES_LIST ?= 1 2 4 8
FF_THREADS ?= 4
RECORDS_SIZE_M ?= 100
PAYLOAD_SIZE_B ?= 64

# Target to run hybrid performance tests with specified MPI nodes and FF threads
# This target will now manage the header and footer for the output table.
# Ensure the target executable is built before running.
test_perf_hybrid: $(TARGET_HYBRID_PERF)
	@echo "=============================================================================="
	@echo "           Scientific MPI Scaling Analysis - Hybrid MergeSort                 "
	@echo "------------------------------------------------------------------------------"
	@echo " Config: $(RECORDS_SIZE_M)M records, $(PAYLOAD_SIZE_B)B payload, $(FF_THREADS) parallel threads/process"
	@echo "=============================================================================="
	@echo "MPI Procs   Time (ms)      Throughput (MRec/s)   Speedup      Efficiency (%)"
	@echo "----------- -------------- ------------------- ------------ ---------------"
	@rm -f hybrid_performance_results.csv # Remove existing CSV to ensure fresh start
	@$(foreach N,$(MPI_NODES_LIST), \
	    mpirun --oversubscribe -np $(N) $(TARGET_HYBRID_PERF) $(FF_THREADS) .baseline_time_hybrid_$(FF_THREADS).tmp $(RECORDS_SIZE_M) $(PAYLOAD_SIZE_B) hybrid_performance_results.csv; \
	)
	@echo "------------------------------------------------------------------------------"
	@echo " Analysis Complete. Note: Speedup and Efficiency are relative to 1 MPI process."
	@echo " Results saved to: hybrid_performance_results.csv"
	@echo "=============================================================================="
	@rm -f .baseline_time_hybrid_$(FF_THREADS).tmp # Clean up the temporary baseline file

# Example of how to run with different parameters:
# make test_perf_hybrid MPI_NODES_LIST="1 2 3" FF_THREADS=2
# make test_perf_hybrid FF_THREADS=8
# make test_perf_hybrid MPI_NODES_LIST="1"
