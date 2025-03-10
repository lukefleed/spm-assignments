CXX                = g++ -std=c++17
OPTFLAGS	   = -O3 -ffast-math -fopenmp
# Default configuration
PARALLEL          ?= 1
USE_AVX512        ?= 1
# Define flags based on configuration
ifeq ($(USE_AVX512),1)
  AVX_FLAG = -march=native
else
  AVX_FLAG = -mavx2
endif
AUTOFLAGS = $(AVX_FLAG) -ftree-vectorize -fopenmp
AVXFLAGS = -mavx2 -mfma -malign-double -falign-loops=32 -fopenmp
CXXFLAGS          += -Wall -Wextra -fno-math-errno
INCLUDES	   = -I. -I./include
LIBS               = -lpthread

TEST_SRC = softmax_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
OBJS = softmax_plain.o softmax_auto.o softmax_avx.o

.PHONY: all clean cleanall test test-parallel-avx512 test-parallel-noavx512 test-noparallel-avx512 test-noparallel-noavx512 test-all

all: softmax_plain softmax_auto softmax_avx

# Uses AVX512 and parallelization
test: softmax_test
	./softmax_test

test-all:
	@echo "Running test-parallel-avx512..."
	$(MAKE) test-parallel-avx512
	@echo "Running test-parallel-noavx512..."
	$(MAKE) test-parallel-noavx512
	@echo "Running test-noparallel-avx512..."
	$(MAKE) test-noparallel-avx512
	@echo "Running test-noparallel-noavx512..."
	$(MAKE) test-noparallel-noavx512

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

softmax_test: $(OBJS) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $^ $(LIBS)

softmax_plain.o: softmax_plain.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -DTEST_BUILD $(INCLUDES) -c $< -o $@

softmax_auto.o: softmax_auto.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(AUTOFLAGS) -DPARALLEL=$(PARALLEL) -DUSE_AVX512=$(USE_AVX512) -DTEST_BUILD $(INCLUDES) -c $< -o $@

softmax_avx.o: softmax_avx.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(AVXFLAGS) -DTEST_BUILD $(INCLUDES) -c $< -o $@

$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -DPARALLEL=$(PARALLEL) -DUSE_AVX512=$(USE_AVX512) $(INCLUDES) -c $< -o $@

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

%_avx: CXXFLAGS += ${AVXFLAGS}
%_auto: CXXFLAGS += ${AUTOFLAGS}

clean:
	-rm -fr *.o *~ softmax_test

cleanall: clean
	-rm -fr softmax_plain softmax_auto softmax_avx
