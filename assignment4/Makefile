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
HYBRID_SRCS = $(SRC_DIR)/hybrid/mpi_communication.cpp $(SRC_DIR)/hybrid/mpi_ff_mergesort.cpp
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
.PHONY: all clean test install single_node multi_node test_correctness_single_node test_perf_single_node

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
# Single node correctness test
test_correctness_single_node: $(BUILD_DIR)/test_correctness
	@echo "Running single node correctness tests..."
	$(BUILD_DIR)/test_correctness

# Single node performance test
test_perf_single_node: $(BUILD_DIR)/test_performance
	@echo "Running single node performance benchmarks..."
	@echo "This may take several minutes to complete..."
	$(BUILD_DIR)/test_performance

# Install (copy to root)
install: all
	cp $(BUILD_DIR)/single_node_sort ./
	cp $(BUILD_DIR)/multi_node_sort ./

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
	@echo "  test                         - Build test executables"
	@echo "  test_correctness_single_node - Run comprehensive correctness tests"
	@echo "  test_perf_single_node        - Run HPC performance benchmarks (saves CSV)"
	@echo "  install                      - Copy executables to project root"
	@echo "  clean                        - Remove build artifacts"
	@echo "  help                         - Show this help message"
