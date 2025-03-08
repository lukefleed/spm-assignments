CXX                = g++ -std=c++17
OPTFLAGS	   = -O3 -ffast-math -fopenmp
AUTOFLAGS = -march=native -ftree-vectorize -fopenmp
AVXFLAGS = -mavx2 -mfma -malign-double -falign-loops=32
CXXFLAGS          += -Wall -Wextra -fno-math-errno
INCLUDES	   = -I. -I./include
LIBS               = -lpthread

# Nuova sezione per la test-suite
TEST_SRC = softmax_test.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
OBJS = softmax_plain.o softmax_auto.o softmax_avx.o

# Mantieni i target originali come PHONY
.PHONY: all clean cleanall test

all: softmax_plain softmax_auto softmax_avx

test: softmax_test
	./softmax_test

softmax_test: $(OBJS) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $^ $(LIBS)

# Regole di compilazione specifiche
softmax_plain.o: softmax_plain.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize $(INCLUDES) -c $< -o $@

softmax_auto.o: softmax_auto.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(AUTOFLAGS) $(INCLUDES) -c $< -o $@

softmax_avx.o: softmax_avx.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(AVXFLAGS) $(INCLUDES) -c $< -o $@

$(TEST_OBJ): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) -c $< -o $@

# Mantieni le regole originali per gli eseguibili singoli
%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

%_avx: CXXFLAGS += ${AVXFLAGS}
%_auto: CXXFLAGS += ${AUTOFLAGS}

clean:
	-rm -fr *.o *~ softmax_test

cleanall: clean
	-rm -fr $(TARGET)
