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
AUTOFLAGS = $(if $(filter 1,$(USE_AVX512)),-march=native,-mavx2) -ftree-vectorize -fopenmp
AVXFLAGS = -mavx2 -mfma -malign-double -falign-loops=32 -fopenmp
CXXFLAGS	+= -Wall -Wextra -fno-math-errno
INCLUDES	= -I. -I./include
LIBS	= -lpthread

TEST_SRC = softmax_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
OBJS = softmax_plain.o softmax_auto.o softmax_avx.o

.PHONY: all clean cleanall test test-parallel-avx512 test-parallel-noavx512 test-noparallel-avx512 test-noparallel-noavx512 test-all test-scaling test-performance test-stability

all: softmax_plain softmax_auto softmax_avx

# Rules for building object files
softmax_plain.o: softmax_plain.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(INCLUDES) -DTEST_BUILD -c $< -o $@

softmax_auto.o: softmax_auto.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AUTOFLAGS) $(INCLUDES) -DTEST_BUILD -c $< -o $@

softmax_avx.o: softmax_avx.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(AVXFLAGS) $(INCLUDES) -DTEST_BUILD -c $< -o $@

# Generic rule for other .cpp files
%.o: %.cpp
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rules for building executables
softmax_plain: softmax_plain.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_auto: softmax_auto.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_avx: softmax_avx.o
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

softmax_test: $(TEST_OBJ) $(OBJS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Uses AVX512 and parallelization
test: softmax_test
	./softmax_test

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
	$(MAKE) test-parallel-avx512
	@echo "Running test-parallel-noavx512..."
	$(MAKE) test-parallel-noavx512
	@echo "Running test-noparallel-avx512..."
	$(MAKE) test-noparallel-avx512
	@echo "Running test-noparallel-noavx512..."
	$(MAKE) test-noparallel-noavx512

# Stability test with AVX512 and parallel execution
test-stability: PARALLEL=1
test-stability: USE_AVX512=1
test-stability: clean softmax_test
	./softmax_test --stability-test

# Target for thread scaling with AVX512 only
test-scaling: PARALLEL=1
test-scaling: USE_AVX512=1
test-scaling: clean softmax_test
	./softmax_test --thread-scaling

# Configuration-specific test targets
test-parallel-avx512: PARALLEL=1
test-parallel-avx512: USE_AVX512=1
test-parallel-avx512: clean softmax_test
	./softmax_test

test-parallel-noavx512: PARALLEL=1
test-parallel-noavx512: USE_AVX512=0
test-parallel-noavx512: clean softmax_test
	./softmax_test

test-noparallel-avx512: PARALLEL=0
test-noparallel-avx512: USE_AVX512=1
test-noparallel-avx512: clean softmax_test
	./softmax_test

test-noparallel-noavx512: PARALLEL=0
test-noparallel-noavx512: USE_AVX512=0
test-noparallel-noavx512: clean softmax_test
	./softmax_test
