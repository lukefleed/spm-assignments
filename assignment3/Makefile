# Compiler and Standard
CXX := g++
CPP_STANDARD := -std=c++17

# Base Include Paths
INCLUDES := -Isrc/app -Isrc/core -Isrc/utils -Iminiz

# --- Compiler Flags ---
CXXFLAGS := -Wall -Wextra -pedantic $(CPP_STANDARD)
OPTFLAGS := -O3 -DNDEBUG
DBGFLAGS := -g -O0
OMPFLAGS := -fopenmp

# --- Linker Flags ---
LDFLAGS_BASE :=
LDFLAGS_OMP := -fopenmp

# --- Executable Names ---
TARGET_APP_PAR := minizp
TARGET_APP_SEQ := minizp_seq
TARGET_TEST := minizp_test
TARGET_BENCH := minizp_bench

# --- Source Files ---
# Core sources used by all targets (excluding main files)
CORE_SRCS := $(wildcard src/core/*.cpp) $(wildcard src/utils/*.cpp) src/app/config.cpp miniz/miniz.c
# Main sources for each executable
APP_MAIN_SRC := src/app/minizp.cpp
TEST_MAIN_SRC := src/app/test_main.cpp
BENCH_MAIN_SRC := src/app/bench_main.cpp

# --- Object Directories ---
OBJDIR_PAR_RELEASE := obj/par/release
OBJDIR_SEQ_RELEASE := obj/seq/release
# Add DBG dirs if needed

# --- Object File Generation ---
# Core Objects (Parallel Release Example)
CORE_OBJS_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(filter src/%.cpp, $(CORE_SRCS))) \
                         $(patsubst miniz/%.c,$(OBJDIR_PAR_RELEASE)/miniz.o, $(filter miniz/%.c, $(CORE_SRCS)))
# Main Object Files (Parallel Release Example)
APP_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(APP_MAIN_SRC))
TEST_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(TEST_MAIN_SRC))
BENCH_MAIN_OBJ_PAR_RELEASE := $(patsubst src/%.cpp,$(OBJDIR_PAR_RELEASE)/%.o, $(BENCH_MAIN_SRC))

# --- Core Object Compilation Rule (Parallel Release Example) ---
$(OBJDIR_PAR_RELEASE)/%.o: src/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling Core $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR_PAR_RELEASE)/miniz.o: miniz/miniz.c
	@mkdir -p $(@D)
	@echo "Compiling Miniz $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# --- Main Object Compilation Rule (Parallel Release Example) ---
$(OBJDIR_PAR_RELEASE)/app/%.o: src/app/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling Main $< -> $@ (Parallel Release)..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# Add similar rules for Sequential and Debug builds...

# --- Linking Rules (Parallel Release Example) ---
$(TARGET_APP_PAR): $(APP_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (App Parallel Release)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

$(TARGET_TEST): $(TEST_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (Test Executable)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

$(TARGET_BENCH): $(BENCH_MAIN_OBJ_PAR_RELEASE) $(CORE_OBJS_PAR_RELEASE)
	@echo "Linking $@ (Benchmark Executable)..."
	$(CXX) $(LDFLAGS_BASE) $(LDFLAGS_OMP) $^ -o $@

# Add similar linking rules for Sequential and Debug targets...

# --- Build Targets ---
.PHONY: all app test_exec bench_exec test bench clean cleanall

all: app # Default target builds the main app

app: $(TARGET_APP_PAR) # Build parallel release by default
app_seq: $(TARGET_APP_SEQ) # Specific target for sequential app

test_exec: $(TARGET_TEST) # Target to only build the test executable
bench_exec: $(TARGET_BENCH) # Target to only build the benchmark executable

# --- Execution Targets ---
test: test_exec
	@echo "Running Correctness Tests (Sequential Mode)..."
	@./$(TARGET_TEST) --mode=seq
	@echo ""
	@echo "Running Correctness Tests (Parallel Mode)..."
	@./$(TARGET_TEST) --mode=par
	@echo ""
	@echo "Correctness Tests Finished."

bench: bench_exec
	@echo "Running Benchmarks..."
	./$(TARGET_BENCH) --type=$(or $(TYPE),one_large) \
	                  --threads=$(or $(THREADS),$(shell nproc)) \
	                  --iterations=$(or $(ITERS),2) \
	                  --warmup=$(or $(WARMUP),1) \

# --- Clean Targets ---
clean:
	@echo "Cleaning object files and executables..."
	rm -f $(TARGET_APP_PAR) $(TARGET_APP_SEQ) $(TARGET_TEST) $(TARGET_BENCH) # Add debug targets if they exist
	rm -rf obj/

cleanall: clean
	@echo "Cleaning generated test/bench data..."
	rm -rf test_data_correctness_cpp/ test_data_bench_cpp/
