CXX	= g++ -std=c++17
OPTFLAGS	= -O3 -ffast-math -fopenmp
# Default configuration
PARALLEL	?= 1
USE_AVX512	?= 1
# Define flags based on configuration
ifeq ($(USE_AVX512),1)
	AVX_FLAG = -march=native
	LOOP_ALIGN = 64
else
	AVX_FLAG = -mavx2
	LOOP_ALIGN = 32
endif
AUTOFLAGS = $(AVX_FLAG) -ftree-vectorize -fopenmp
AVXFLAGS = -mavx2 -mfma -malign-double -falign-loops=$(LOOP_ALIGN) -fopenmp
CXXFLAGS	+= -Wall -fno-math-errno
INCLUDES	= -I. -I./include
LIBS	= -lpthread

TEST_SRC = softmax_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
OBJS = softmax_plain.o softmax_auto.o softmax_avx.o

# Object files for standalone executables
STANDALONE_OBJS = standalone_softmax_plain.o standalone_softmax_auto.o standalone_softmax_avx.o

.PHONY: all clean cleanall test test-parallel-avx512 test-parallel-noavx512 test-noparallel-avx512 test-noparallel-noavx512 test-all test-scaling test-performance test-stability help

# Default target shows help
.DEFAULT_GOAL := help

help:
	@echo "======================================================================="
	@echo "                    SOFTMAX OPTIMIZATION MAKEFILE                      "
	@echo "======================================================================="
	@echo ""
	@echo "This Makefile builds and tests three softmax implementations:"
	@echo "  1. Plain (scalar) implementation"
	@echo "  2. Auto-vectorized implementation (compiler-driven)"
	@echo "  3. Manually vectorized implementation using AVX intrinsics"
	@echo ""
	@echo "CONFIGURATION VARIABLES:"
	@echo "------------------------"
	@echo "  PARALLEL     : Enable/disable OpenMP parallelization (0/1, default: 1)"
	@echo "  USE_AVX512   : Use AVX512 instructions if available (0/1, default: 1)"
	@echo "                 0 = use -mavx2, 1 = use -march=native"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "--------------"
	@echo "  all          : Build all three standalone executables"
	@echo "  softmax_plain: Build plain scalar implementation"
	@echo "  softmax_auto : Build auto-vectorized implementation"
	@echo "  softmax_avx  : Build manual AVX implementation"
	@echo "  softmax_test : Build test harness"
	@echo ""
	@echo "TEST TARGETS:"
	@echo "-------------"
	@echo "  test         : Quick performance test (AVX512 + parallel)"
	@echo "  test-all     : Run all test types (performance, stability, scaling)"
	@echo ""
	@echo "  Performance Tests:"
	@echo "    test-performance           : Run all 4 performance configurations"
	@echo "    test-parallel-avx512       : Parallel + AVX512"
	@echo "    test-parallel-noavx512     : Parallel + AVX2"
	@echo "    test-noparallel-avx512     : Sequential + AVX512"
	@echo "    test-noparallel-noavx512   : Sequential + AVX2"
	@echo ""
	@echo "  Analysis Tests:"
	@echo "    test-stability : Numerical stability analysis"
	@echo "    test-scaling   : Thread scaling analysis"
	@echo ""
	@echo "CLEANING:"
	@echo "---------"
	@echo "  clean        : Remove object files and executables"
	@echo "  cleanall     : Remove all generated files including test objects"
	@echo ""
	@echo "USAGE EXAMPLES:"
	@echo "---------------"
	@echo "  # Build with default settings (parallel, AVX512)"
	@echo "  make all"
	@echo ""
	@echo "  # Build for AVX2 only (no AVX512)"
	@echo "  make USE_AVX512=0 all"
	@echo ""
	@echo "  # Build sequential version"
	@echo "  make PARALLEL=0 all"
	@echo ""
	@echo "  # Run quick performance test"
	@echo "  make test"
	@echo ""
	@echo "  # Run complete test suite"
	@echo "  make test-all"
	@echo ""
	@echo "  # Test specific configuration"
	@echo "  make PARALLEL=0 USE_AVX512=1 test-noparallel-avx512"
	@echo ""
	@echo "OUTPUT FILES:"
	@echo "-------------"
	@echo "  results/performance/       : Performance benchmark CSV files"
	@echo "  results/speedup/           : Speedup analysis CSV files"
	@echo "  results/numerical_stability/: Stability test CSV files"
	@echo "  results/thread_scaling/    : Thread scaling CSV files"
	@echo "  images/                    : Generated plots and visualizations"
	@echo ""
	@echo "COMPILER FLAGS:"
	@echo "---------------"
	@echo "  Base flags: -std=c++17 -O3 -ffast-math -fopenmp -Wall -fno-math-errno"
	@echo "  Auto-vec:   + -ftree-vectorize + AVX_FLAG"
	@echo "  Manual AVX: + -mavx2 -mfma -malign-double -falign-loops=32"
	@echo ""
	@echo "======================================================================="

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
