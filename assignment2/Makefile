CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O3 -g -pthread
LDFLAGS = -pthread
BUILD_DIR = build

EXE = collatz_par
EXECUTABLE = $(BUILD_DIR)/$(EXE)

# Source files
SRCS = src/main.cpp \
       src/sequential.cpp \
       src/static_scheduler.cpp \
       src/dynamic_scheduler.cpp \
       src/collatz.cpp \
       src/testing.cpp \
       src/utils.cpp \
	   src/theoretical_analysis.cpp

# Object files: updated pattern to match src/ directory structure
OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/obj/%.o, $(SRCS))

# Default target: build the executable
all: $(EXECUTABLE)

# Rule to link the executable
$(EXECUTABLE): $(OBJS)
	@echo "Linking executable..."
	@mkdir -p $(@D) # Create build directory if it doesn't exist (@D is the directory part of $@)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "Executable created: $@"

# Rule to compile .cpp files into .o files in the build directory
$(BUILD_DIR)/obj/%.o: src/%.cpp
	@echo "Compiling $<..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

# Target to clean build artifacts
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)

# --- Help Target ---

# Target to display comprehensive help information
help:
	@echo "=================================================="
	@echo "           COLLATZ CONJECTURE PROJECT"
	@echo "=================================================="
	@echo ""
	@echo "MAKE TARGETS:"
	@echo "  all              - Build the executable (default target)"
	@echo "  clean            - Remove all build artifacts"
	@echo "  test             - Run correctness tests (alias for test_correctness)"
	@echo "  test_correctness - Run the correctness test suite"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  theory           - Run theoretical analysis"
	@echo "  help             - Display this help message"
	@echo ""
	@echo "EXECUTABLE USAGE:"
	@echo "  ./$(EXECUTABLE) [options] range1 [range2] ..."
	@echo ""
	@echo "  Arguments:"
	@echo "    rangeN               Required. One or more ranges in format start-end (e.g., 1-1000)"
	@echo ""
	@echo "  Options:"
	@echo "    -s <variant>         Static scheduling: block, cyclic, block-cyclic (default: block-cyclic)"
	@echo "    -d                   Use dynamic work stealing scheduling"
	@echo "    -n <threads>         Number of threads for parallel execution (default: 1, sequential)"
	@echo "    -c <size>            Chunk/block size for scheduling (default: 64)"
	@echo "    -v                   Enable verbose output"
	@echo "    -t, --theory         Run theoretical analysis (saves to results/theoretical_speedup.csv)"
	@echo "    -h, --help           Show executable help message"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  Build and test:"
	@echo "    make && make test"
	@echo ""
	@echo "  Sequential execution:"
	@echo "    ./$(EXECUTABLE) 1-10000"
	@echo ""
	@echo "  Static scheduling variants:"
	@echo "    ./$(EXECUTABLE) -n 8 1-1000000                    # Block-cyclic (default)"
	@echo "    ./$(EXECUTABLE) -s block -n 4 1-1k 10k-20k        # Block scheduling"
	@echo "    ./$(EXECUTABLE) -s cyclic -n 4 1-100000           # Cyclic scheduling"
	@echo ""
	@echo "  Dynamic scheduling:"
	@echo "    ./$(EXECUTABLE) -d -n 16 -c 1024 1-1M             # 16 threads, chunk size 1024"
	@echo ""
	@echo "  Theoretical analysis:"
	@echo "    ./$(EXECUTABLE) --theory"
	@echo ""
	@echo "  Full benchmark suite:"
	@echo "    make benchmark"
	@echo ""
	@echo "=================================================="

# --- Testing and Benchmarking Targets ---

# Target to run the correctness test suite
test_correctness: all
	@echo "---------------------------"
	@echo " Running Correctness Tests "
	@echo "---------------------------"
	./$(EXECUTABLE) --test-correctness

# Target to run the performance benchmark suite
benchmark: all
	@echo "----------------------------------"
	@echo " Running Performance Benchmarks "
	@echo "----------------------------------"
	./$(EXECUTABLE) --benchmark

theory:
	@echo "----------------------------------"
	@echo " Running Theoretical Analysis "
	@echo "----------------------------------"
	./$(EXECUTABLE) --theory

# Convenience target 'test' runs correctness tests by default
test: test_correctness

# Declare targets that are not files
.PHONY: all clean test_correctness benchmark test help help
