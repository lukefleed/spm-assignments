# Makefile for Collatz Conjecture Parallel Implementation

# Compiler and flags
CXX = g++
# C++ Standard: C++17 required for std::filesystem, std::string_view
# Optimization: -O3 for performance benchmarks
# Debugging: -g includes debug symbols
# Warnings: -Wall -Wextra -pedantic enable most useful warnings
# Threading: -pthread needed for std::thread
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O3 -g -pthread
LDFLAGS = -pthread
# LIBS = -lstdc++fs # Usually not needed with modern g++/clang++ and -std=c++17, uncomment if linking filesystem fails

# Build directory
BUILD_DIR = build

# Executable name
EXE = collatz_par
EXECUTABLE = $(BUILD_DIR)/$(EXE)

# Source files
SRCS = src/main.cpp \
       src/sequential.cpp \
       src/static_scheduler.cpp \
       src/dynamic_scheduler.cpp \
       src/collatz.cpp \
       src/testing.cpp \
       src/utils.cpp

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

# Convenience target 'test' runs correctness tests by default
test: test_correctness

# Declare targets that are not files
.PHONY: all clean test_correctness benchmark test

# Optional: Improve dependency tracking (recompile only if headers change)
# Uncomment the following lines to enable automatic header dependency generation
# DEPS = $(OBJS:.o=.d)
# CXXFLAGS += -MMD -MP
# -include $(DEPS)
