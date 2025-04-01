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
.PHONY: all clean test_correctness benchmark test
