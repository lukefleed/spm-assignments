#------------------------------------------------------------------------------
# Softmax Function Optimization Makefile
#------------------------------------------------------------------------------
#
# This Makefile builds and tests three different implementations of the softmax
# function:
#   1. Plain (scalar) implementation
#   2. Auto-vectorized implementation (compiler-driven)
#   3. Manually vectorized implementation using AVX intrinsics
#
# COMPILER FLAGS EXPLANATION:
# --------------------------
# - std=c++17           : C++17 standard
# - -O3                 : Highest level of optimization for best performance
# - -ffast-math         : Relaxes IEEE compliance for better FP performance
# - -fopenmp            : Enables OpenMP for parallelization
# - -Wall -Wextra       : Comprehensive warning messages for code quality
# - -fno-math-errno     : Avoids setting errno after math operations
# VECTORIZATION STRATEGY:
# ----------------------
# Auto-vectorized implementation:
#   - Uses compiler flags to enable automatic SIMD optimization
#   - -ftree-vectorize  : Explicitly requests vectorization
#   - AVX_FLAG          : Either -march=native (for AVX512) or -mavx2
#
# Manual AVX implementation:
#   - -mavx2            : Enables AVX2 instruction set
#   - -mfma             : Enables fused multiply-add operations
#   - -malign-double    : Improves alignment of doubles for vectorization
#   - -falign-loops=32  : Aligns loops to 32 bytes for better cache usage
#
# CONFIGURATION VARIABLES:
# ----------------------
# PARALLEL (0/1)      : Controls OpenMP parallelization
#                       Set to 0 for sequential execution, 1 for parallel
#
# USE_AVX512 (0/1)    : Controls instruction set targeting
#                       Set to 1 for -march=native (AVX512 if available)
#                       Set to 0 for explicit -mavx2 targeting
#
# BUILD PROCESS:
# -------------
# - Separate object files are created for testing vs. standalone use
# - Test objects include -DTEST_BUILD for integration with test harness
# - Standalone objects allow individual execution of each implementation
#
# TEST CONFIGURATIONS:
# ------------------
# The Makefile supports multiple test configurations to analyze performance:
# - Parallel vs. Sequential execution
# - AVX512 vs. AVX2 instruction sets
# - Performance benchmarks, numerical stability tests, thread scaling tests
#
# The test results are saved to CSV files under results/ and visualized
# with Python scripts (plot.py and plot_scaling.py).
#------------------------------------------------------------------------------

CXX	= g++ -std=c++17
OPTFLAGS	= -O3 -ffast-math -fopenmp
# Default configuration
PARALLEL	?= 1
USE_AVX512	?= 1
# Define flags based on configuration
ifeq ($(USE_AVX512),1)
	AVX_FLAG = -march=native
else
	AVX_FLAG = -mavx2
endif
AUTOFLAGS = $(AVX_FLAG) -ftree-vectorize -fopenmp -fopt-info-vec-missed
AVXFLAGS = -mavx2 -mfma -malign-double -falign-loops=32 -fopenmp
CXXFLAGS	+= -Wall -fno-math-errno
INCLUDES	= -I. -I./include
LIBS	= -lpthread

TEST_SRC = softmax_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
OBJS = softmax_plain.o softmax_auto.o softmax_avx.o

# Object files for standalone executables
STANDALONE_OBJS = standalone_softmax_plain.o standalone_softmax_auto.o standalone_softmax_avx.o

.PHONY: all clean cleanall test test-parallel-avx512 test-parallel-noavx512 test-noparallel-avx512 test-noparallel-noavx512 test-all test-scaling test-performance test-stability

all: softmax_plain softmax_auto softmax_avx

# Rules for building test object files (with TEST_BUILD defined)
softmax_plain.o: softmax_plain.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(INCLUDES) -DTEST_BUILD -c $< -o $@

softmax_auto.o: softmax_auto.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AUTOFLAGS) $(INCLUDES) -DTEST_BUILD -DPARALLEL=$(PARALLEL) -DUSE_AVX512=$(USE_AVX512) -c $< -o $@

softmax_avx.o: softmax_avx.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AVXFLAGS) $(INCLUDES) -DTEST_BUILD -c $< -o $@

# Rules for building standalone object files (without TEST_BUILD)
standalone_softmax_plain.o: softmax_plain.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

standalone_softmax_auto.o: softmax_auto.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AUTOFLAGS) $(INCLUDES) -DPARALLEL=$(PARALLEL) -DUSE_AVX512=$(USE_AVX512) -c $< -o $@

standalone_softmax_avx.o: softmax_avx.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AVXFLAGS) $(INCLUDES) -c $< -o $@

# Rules for building test object files (without TEST_BUILD)
%.o: %.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(INCLUDES) -DPARALLEL=$(PARALLEL) -DUSE_AVX512=$(USE_AVX512) -c $< -o $@

# Rules for building executables using the standalone object files
softmax_plain: standalone_softmax_plain.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_auto: standalone_softmax_auto.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_avx: standalone_softmax_avx.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_test: $(TEST_OBJ) $(OBJS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Uses AVX512 and parallelization
test: softmax_test
	./softmax_test --performance-only

# Clean build files
clean:
	rm -f *.o softmax_plain softmax_auto softmax_avx softmax_test

# Clean all generated files
cleanall: clean
	rm -f $(TEST_OBJ)

# Run all test types (performance, stability, scaling)
test-all:
	@echo "Running performance tests..."
	$(MAKE) test-performance
	@echo "Running stability test..."
	$(MAKE) test-stability
	@echo "Running thread scaling test..."
	$(MAKE) test-scaling

# Run all configurations for performance testing
test-performance:
	@echo "Running test-parallel-avx512..."
	$(MAKE) -e PARALLEL=1 USE_AVX512=1 test-parallel-avx512
	@echo "Running test-parallel-noavx512..."
	$(MAKE) -e PARALLEL=1 USE_AVX512=0 test-parallel-noavx512
	@echo "Running test-noparallel-avx512..."
	$(MAKE) -e PARALLEL=0 USE_AVX512=1 test-noparallel-avx512
	@echo "Running test-noparallel-noavx512..."
	$(MAKE) -e PARALLEL=0 USE_AVX512=0 test-noparallel-noavx512

# Stability test with AVX512 and parallel execution
test-stability:
	$(MAKE) -e PARALLEL=1 USE_AVX512=1 clean softmax_test
	./softmax_test --stability-test

# Target for thread scaling with AVX512 only
test-scaling:
	$(MAKE) -e PARALLEL=1 USE_AVX512=1 clean softmax_test
	./softmax_test --thread-scaling

# Configuration-specific test targets
test-parallel-avx512:
	$(MAKE) -e PARALLEL=1 USE_AVX512=1 clean softmax_test
	./softmax_test --performance-only

test-parallel-noavx512:
	$(MAKE) -e PARALLEL=1 USE_AVX512=0 clean softmax_test
	./softmax_test --performance-only

test-noparallel-avx512:
	$(MAKE) -e PARALLEL=0 USE_AVX512=1 clean softmax_test
	./softmax_test --performance-only

test-noparallel-noavx512:
	$(MAKE) -e PARALLEL=0 USE_AVX512=0 clean softmax_test
	./softmax_test --performance-only
