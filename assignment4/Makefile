# Compiler and flags
CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread -I$(INCLUDE_DIR) -I$(FF_DIR)
LDFLAGS = -pthread
LIBS =

# Directories
SRC_DIR = src
BUILD_DIR = bin
OBJ_DIR = obj
INCLUDE_DIR = include
FF_DIR = fastflow

# Source files
COMMON_SRCS = $(SRC_DIR)/common/utils.cpp
SEQ_SRCS = $(SRC_DIR)/sequential/sequential_mergesort.cpp
FF_MERGESORT_SRC = $(SRC_DIR)/fastflow/ff_mergesort.cpp
# MODIFIED: Removed mpi_communication.cpp as its logic is now integrated
#           into mpi_ff_mergesort.cpp for safety and efficiency.
HYBRID_SRCS = $(SRC_DIR)/hybrid/mpi_ff_mergesort.cpp
SINGLE_NODE_MAIN = $(SRC_DIR)/main/single_node_main.cpp
MULTI_NODE_MAIN = $(SRC_DIR)/main/multi_node_main.cpp

# Object files
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))
SEQ_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SEQ_SRCS))
FF_MERGESORT_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_MERGESORT_SRC))
HYBRID_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_SRCS))

# Test files
TESTS_DIR = tests
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.cpp)
TEST_BINS = $(patsubst $(TESTS_DIR)/%.cpp,$(BUILD_DIR)/test_%,$(TEST_SRCS))

# Main targets
.PHONY: all clean test install single_node multi_node test_correctness_single_node test_perf_single_node debug_mpi debug_deep

all: single_node multi_node

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/common $(OBJ_DIR)/sequential $(OBJ_DIR)/fastflow $(OBJ_DIR)/hybrid $(OBJ_DIR)/main

# Single node executable
single_node: $(BUILD_DIR)/single_node_sort

$(BUILD_DIR)/single_node_sort: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(SINGLE_NODE_MAIN)
	$(CXX) $(CXXFLAGS) -o $@ $(SINGLE_NODE_MAIN) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(LDFLAGS) $(LIBS)

# Multi node executable
multi_node: $(BUILD_DIR)/multi_node_sort

$(BUILD_DIR)/multi_node_sort: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(MULTI_NODE_MAIN)
	$(MPICXX) $(CXXFLAGS) -o $@ $(MULTI_NODE_MAIN) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

# Object file compilation rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule for hybrid (MPI) source files
$(OBJ_DIR)/hybrid/%.o: $(SRC_DIR)/hybrid/%.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Test target for the main implementation
$(BUILD_DIR)/ff_test: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(FF_MERGESORT_OBJ)
	$(CXX) $(CXXFLAGS) -DTEST_MAIN -o $@ $(FF_MERGESORT_SRC) $(COMMON_OBJS) $(LDFLAGS) $(LIBS)

# Test targets
test: $(TEST_BINS)

$(BUILD_DIR)/test_%: $(TESTS_DIR)/%.cpp $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(LDFLAGS) $(LIBS)

# Specific rule for correctness test
$(BUILD_DIR)/test_correctness: $(TESTS_DIR)/test_correctness.cpp $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(LDFLAGS) $(LIBS)

# Specific rule for performance test
$(BUILD_DIR)/test_performance: $(TESTS_DIR)/test_performance.cpp $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(LDFLAGS) $(LIBS)

# Single node correctness test
test_correctness_single_node: $(BUILD_DIR)/test_correctness
	@echo "Running single node correctness tests..."
	$(BUILD_DIR)/test_correctness

# Single node performance test
test_perf_single_node: $(BUILD_DIR)/test_performance
	@echo "Running single node performance benchmarks..."
	@echo "This may take several minutes to complete..."
	$(BUILD_DIR)/test_performance

# Hybrid correctness test
test_correctness_hybrid: $(BUILD_DIR)/test_hybrid_correctness
	@echo "Running hybrid MPI+FastFlow correctness tests..."
	mpirun -np 2 $(BUILD_DIR)/test_hybrid_correctness

# Hybrid performance test - Scientific MPI scaling analysis
test_perf_hybrid: $(BUILD_DIR)/test_hybrid_performance
	@echo "================================================================"
	@echo "Scientific MPI Scaling Analysis - Hybrid MPI+FastFlow Mergesort"
	@echo "Fixed: 10M records, 64B payload, 4 FastFlow threads/process"
	@echo "================================================================"
	@echo ""
	@echo "Testing with 1 MPI process (baseline)..."
	@echo "----------------------------------------"
	mpirun -np 1 $(BUILD_DIR)/test_hybrid_performance
	@echo ""
	@echo "Testing with 2 MPI processes..."
	@echo "-------------------------------"
	mpirun -np 2 $(BUILD_DIR)/test_hybrid_performance
	@echo ""
	@echo "Testing with 3 MPI processes..."
	@echo "-------------------------------"
	mpirun -np 3 $(BUILD_DIR)/test_hybrid_performance
	@echo ""
	@echo "================================================================"
	@echo "MPI Scaling Analysis Complete"
	@echo "================================================================"

# Install (copy to root)
install: all
	cp $(BUILD_DIR)/single_node_sort ./
	cp $(BUILD_DIR)/multi_node_sort ./

# Debug targets
debug_mpi: $(BUILD_DIR)/debug_mpi

$(BUILD_DIR)/debug_mpi: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/main/debug_mpi.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

debug_deep: $(BUILD_DIR)/debug_deep

$(BUILD_DIR)/debug_deep: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/main/debug_deep.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(OBJ_DIR)
	rm -f single_node_sort multi_node_sort

# Help
help:
	@echo "Available targets:"
	@echo "  all                          - Build all executables"
	@echo "  single_node                  - Build single node version"
	@echo "  multi_node                   - Build multi node version"
	@echo ""
	@echo "Testing targets:"
	@echo "  test_correctness_single_node - Run single node correctness tests"
	@echo "  test_perf_single_node        - Run single node performance benchmarks"
	@echo "  test_correctness_hybrid      - Run hybrid MPI+FastFlow correctness tests"
	@echo "  test_perf_hybrid             - Scientific MPI scaling analysis (1,2,3 processes)"
	@echo ""
	@echo "Scientific Analysis:"
	@echo "  test_perf_hybrid             - Complete MPI scaling study with fixed parameters"
	@echo "                                 (10M records, 64B payload, 4 FF threads/process)"
	@echo ""
	@echo "Utility targets:"
	@echo "  install                      - Copy executables to project root"
	@echo "  clean                        - Remove build artifacts"
	@echo "  help                         - Show this help message"

# MPI Direct test target
test_mpi_direct: $(BUILD_DIR)/test_mpi_direct

$(BUILD_DIR)/test_mpi_direct: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/main/test_mpi_direct.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

# Debug deadlock target
debug_deadlock: $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ debug_deadlock.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

# Hybrid test targets
test_hybrid_correctness: $(BUILD_DIR)/test_hybrid_correctness

$(BUILD_DIR)/test_hybrid_correctness: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(TESTS_DIR)/test_hybrid_correctness.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)

test_hybrid_performance: $(BUILD_DIR)/test_hybrid_performance

$(BUILD_DIR)/test_hybrid_performance: $(BUILD_DIR) $(OBJ_DIR) $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ $(TESTS_DIR)/test_hybrid_performance.cpp $(COMMON_OBJS) $(SEQ_OBJS) $(FF_MERGESORT_OBJ) $(HYBRID_OBJS) $(LDFLAGS) $(LIBS)
