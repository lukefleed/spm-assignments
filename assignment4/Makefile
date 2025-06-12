# CXX and MPI Compilers
CXX = g++
MPICXX = mpic++

# Compiler and Linker Flags
# Added -pthread for parallel framework support
# Common flags for all compilations
COMMON_FLAGS = -std=c++17 -Wall -O3 -I./include -I./fastflow -pthread
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
.PHONY: all clean test_perf_hybrid test_correctness_single_node test_perf_single_node benchmark_single_node benchmark_array_scaling benchmark_payload_scaling test_correctness_hybrid help

# Single node correctness test
test_correctness_single_node: $(TARGET_CORRECT)
	@echo "Running single node correctness tests..."
	$(TARGET_CORRECT)

# Default values for single node performance test
# Generate thread list dynamically: 2, 4, 8, 16, ..., up to max system threads
MAX_THREADS := $(shell nproc)
THREAD_LIST ?= $(shell i=2; while [ $$i -le $(MAX_THREADS) ]; do echo $$i; i=$$((i*2)); done | tr '\n' ' ')
ARRAY_SIZE_M ?= 100
PAYLOAD_SIZE_B ?= 16

# Configurable single node performance test
test_perf_single_node: $(TARGET_PERF)
	@echo "=============================================================================="
	@echo "                   Single Node Performance Analysis                           "
	@echo "------------------------------------------------------------------------------"
	@echo " Config: $(ARRAY_SIZE_M)M records, $(PAYLOAD_SIZE_B)B payload"
	@echo " Threads: $(THREAD_LIST)"
	@echo "=============================================================================="
	$(TARGET_PERF) "$(THREAD_LIST)" $(ARRAY_SIZE_M) $(PAYLOAD_SIZE_B)
	@echo "------------------------------------------------------------------------------"
	@echo " Analysis Complete. Results saved to: performance_results.csv"
	@echo " For scaling analysis, use: make benchmark_array_scaling | benchmark_payload_scaling"
	@echo "=============================================================================="

# Single node performance benchmark with standardized metrics output
SINGLE_NODE_THREADS ?= 8
SINGLE_NODE_OUTPUT ?= results_single_node_performance.csv

benchmark_single_node: $(TARGET_SINGLE_NODE)
	@echo "=============================================================================="
	@echo "                    Single Node Performance Benchmark                        "
	@echo "------------------------------------------------------------------------------"
	@echo " Config: $(ARRAY_SIZE_M)M records, $(PAYLOAD_SIZE_B)B payload, $(SINGLE_NODE_THREADS) threads"
	@echo " Output: $(SINGLE_NODE_OUTPUT)"
	@echo "=============================================================================="
	$(TARGET_SINGLE_NODE) -s $(ARRAY_SIZE_M)M -r $(PAYLOAD_SIZE_B) -t $(SINGLE_NODE_THREADS) --no-validate --csv --csv-file $(SINGLE_NODE_OUTPUT) --verbose
	@echo "------------------------------------------------------------------------------"
	@echo " Benchmark Complete. Results saved to: $(SINGLE_NODE_OUTPUT)"
	@echo " CSV contains standardized performance metrics with proper speedup analysis."
	@echo "=============================================================================="

# Array size scaling benchmark
benchmark_array_scaling: $(TARGET_SINGLE_NODE)
	@echo "Executing array size scaling benchmark..."
	@./benchmark_array_scaling.sh $(SINGLE_NODE_THREADS) benchmark_array_scaling_results.csv

# Payload size scaling benchmark
benchmark_payload_scaling: $(TARGET_SINGLE_NODE)
	@echo "Executing payload size scaling benchmark..."
	@./benchmark_payload_scaling.sh $(SINGLE_NODE_THREADS) benchmark_payload_scaling_results.csv

# Hybrid correctness test
test_correctness_hybrid: $(TARGET_HYBRID_CORRECT)
	@echo "Running hybrid MPI+parallel correctness tests..."
	mpirun --oversubscribe -np 2 $(TARGET_HYBRID_CORRECT)

# Help target
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
	@echo "  test_perf_single_node        - Thread scaling performance analysis (console output)"
	@echo "                                 Usage: make test_perf_single_node THREAD_LIST=\"2 4 8\" ARRAY_SIZE_M=10 PAYLOAD_SIZE_B=64"
	@echo "                                 Outputs: performance_results.csv with detailed metrics"
	@echo "  test_perf_hybrid             - MPI scaling performance analysis with dual-baseline comparison"
	@echo "                                 Usage: make test_perf_hybrid MPI_NODES_LIST=\"1 2 4\" FF_THREADS=4"
	@echo "                                 Outputs: hybrid_performance_results.csv with scaling analysis"
	@echo ""
	@echo "Benchmark Targets:"
	@echo "  benchmark_single_node        - Single configuration comprehensive benchmark"
	@echo "                                 Usage: make benchmark_single_node ARRAY_SIZE_M=50 SINGLE_NODE_THREADS=8"
	@echo "  benchmark_array_scaling      - Array size scaling analysis (200K to 100M records)"
	@echo "                                 Usage: make benchmark_array_scaling SINGLE_NODE_THREADS=4"
	@echo "  benchmark_payload_scaling    - Payload size scaling analysis (2 to 512 bytes)"
	@echo "                                 Usage: make benchmark_payload_scaling SINGLE_NODE_THREADS=6"
	@echo ""
	@echo "Configuration Parameters:"
	@echo ""
	@echo "Single-Node Performance:"
	@echo "  THREAD_LIST=\"2 4 8 16\"       - Thread counts for scaling tests (default: powers of 2 up to nproc)"
	@echo "  SINGLE_NODE_THREADS=8        - Thread count for single configuration benchmarks"
	@echo "  ARRAY_SIZE_M=100             - Dataset size in millions of records"
	@echo "  PAYLOAD_SIZE_B=16            - Record payload size in bytes"
	@echo "  SINGLE_NODE_OUTPUT=file.csv  - Output CSV file for single-node benchmark"
	@echo ""
	@echo "Hybrid MPI Performance:"
	@echo "  MPI_NODES_LIST=\"1 2 4 8\"     - MPI process counts to test"
	@echo "  FF_THREADS=4                 - FastFlow threads per MPI process"
	@echo "  RECORDS_SIZE_M=10            - Dataset size in millions of records"
	@echo "  PAYLOAD_SIZE_B=64            - Record payload size in bytes"
	@echo "  HYBRID_CSV_FILE=results.csv  - Output CSV file for detailed analysis"
	@echo ""
	@echo "Common Usage Examples:"
	@echo ""
	@echo "Quick correctness verification:"
	@echo "  make test_correctness_single_node"
	@echo "  make test_correctness_hybrid"
	@echo ""
	@echo "Thread scaling analysis:"
	@echo "  make test_perf_single_node"
	@echo "  make test_perf_single_node THREAD_LIST=\"1 2 4 8 16\" ARRAY_SIZE_M=50"
	@echo ""
	@echo "Single configuration benchmarks:"
	@echo "  make benchmark_single_node"
	@echo "  make benchmark_single_node ARRAY_SIZE_M=100 SINGLE_NODE_THREADS=12"
	@echo ""
	@echo "Scaling analysis:"
	@echo "  make benchmark_array_scaling SINGLE_NODE_THREADS=8"
	@echo "  make benchmark_payload_scaling SINGLE_NODE_THREADS=4"
	@echo ""
	@echo "MPI hybrid performance:"
	@echo "  make test_perf_hybrid"
	@echo "  make test_perf_hybrid MPI_NODES_LIST=\"1 2 4 8\" FF_THREADS=8 RECORDS_SIZE_M=50"

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
