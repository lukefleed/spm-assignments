# Compiler and flags
CXX = g++
MPICXX = mpicxx
CXX_STD = -std=c++17 # Come da specifiche del corso
OPTIM_FLAGS = -O3 -march=native
DEBUG_FLAGS = -g -Wall -Wextra -Wpedantic
COMMON_FLAGS = $(CXX_STD) $(DEBUG_FLAGS)

# FastFlow specific paths.
FF_HOME = ./fastflow
# FF_INC per includere da $(FF_HOME)/ff/
FF_INC = -I$(FF_HOME)

MPI_INC =
MPI_LIB =

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
SCRIPT_DIR = scripts
DATA_DIR = data

# Common object files
COMMON_SRCS = $(SRC_DIR)/utils.cpp $(SRC_DIR)/performance_timer.cpp $(SRC_DIR)/mergesort_common.cpp
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))

# Single-node (FastFlow) specific
FF_MAIN_SRC = $(SRC_DIR)/main_ff.cpp
FF_MODULE_SRCS = $(SRC_DIR)/mergesort_ff.cpp
FF_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_MODULE_SRCS))
FF_MAIN_OBJ = $(OBJ_DIR)/main_ff.o
EXECUTABLE_FF = $(BIN_DIR)/sort_ff

# Hybrid (MPI + FastFlow) specific
HYBRID_MAIN_SRC = $(SRC_DIR)/main_hybrid.cpp
HYBRID_MODULE_SRCS = $(SRC_DIR)/mergesort_hybrid.cpp $(SRC_DIR)/mergesort_ff.cpp
HYBRID_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_MODULE_SRCS))
HYBRID_MAIN_OBJ = $(OBJ_DIR)/main_hybrid.o
EXECUTABLE_HYBRID = $(BIN_DIR)/sort_hybrid

.PHONY: all clean directories test_correctness_single_node test_correctness_hybrid run_perf_single_node run_perf_hybrid run_all_tests plot_results data

all: directories $(EXECUTABLE_FF) $(EXECUTABLE_HYBRID)

directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(DATA_DIR) $(BUILD_DIR)/plots

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

# Rules for main files (specific FF_INC if needed, but general one should cover)
$(FF_MAIN_OBJ): $(FF_MAIN_SRC) $(INC_DIR)/record.h $(INC_DIR)/utils.h $(INC_DIR)/mergesort_ff.h $(INC_DIR)/performance_timer.h
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

$(HYBRID_MAIN_OBJ): $(HYBRID_MAIN_SRC) $(INC_DIR)/record.h $(INC_DIR)/utils.h $(INC_DIR)/mergesort_hybrid.h $(INC_DIR)/performance_timer.h
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@

# Rules for module files (specific FF_INC if needed)
$(OBJ_DIR)/mergesort_ff.o: $(SRC_DIR)/mergesort_ff.cpp $(INC_DIR)/mergesort_ff.h $(INC_DIR)/record.h $(INC_DIR)/mergesort_common.h
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

$(OBJ_DIR)/mergesort_hybrid.o: $(SRC_DIR)/mergesort_hybrid.cpp $(INC_DIR)/mergesort_hybrid.h $(INC_DIR)/record.h $(INC_DIR)/mergesort_ff.h
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@


$(EXECUTABLE_FF): $(COMMON_OBJS) $(FF_MODULE_OBJS) $(FF_MAIN_OBJ)
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ -lpthread -o $@

$(EXECUTABLE_HYBRID): $(COMMON_OBJS) $(HYBRID_MODULE_OBJS) $(HYBRID_MAIN_OBJ)
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ $(MPI_LIB) -lpthread -o $@

clean:
	rm -rf $(BUILD_DIR)/*
	rm -f results_*.csv
	rm -f $(DATA_DIR)/input_N*_R*.dat

N ?= 1000000
R_PAYLOAD ?= 64
T_THREADS ?= 4
P_PROCS ?= 2

INPUT_FILE_BASE = $(DATA_DIR)/input_N$(N)_R$(R_PAYLOAD)
FF_ARGS = -s $(N) -r $(R_PAYLOAD) -t $(T_THREADS)
HYBRID_ARGS = -s $(N) -r $(R_PAYLOAD) -t $(T_THREADS)

$(DATA_DIR)/input_N%_R%.dat: $(SCRIPT_DIR)/generate_data.py
	@echo "Generating test data: $@"
	python3 $(SCRIPT_DIR)/generate_data.py --size $(*) --payload $(**) --output $@

TARGET_DATA_FILE = $(INPUT_FILE_BASE).dat

test_correctness_single_node: $(EXECUTABLE_FF) $(TARGET_DATA_FILE)
	@echo "Running Correctness Test for Single-Node: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS)"
	./$(EXECUTABLE_FF) $(FF_ARGS) --input $(TARGET_DATA_FILE) --check_correctness

test_correctness_hybrid: $(EXECUTABLE_HYBRID) $(TARGET_DATA_FILE)
	@echo "Running Correctness Test for Hybrid: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS), P=$(P_PROCS)"
	mpirun -np $(P_PROCS) --oversubscribe ./$(EXECUTABLE_HYBRID) $(HYBRID_ARGS) --input $(TARGET_DATA_FILE) --check_correctness

N_VALUES_LIST ?= "10000000 100000000"
R_VALUES_LIST ?= "8 64 256"
T_VALUES_LIST ?= "1 2 4 8 16 32"
P_VALUES_LIST ?= "1 2 4 8"

run_perf_single_node: $(EXECUTABLE_FF) $(SCRIPT_DIR)/run_benchmarks.sh
	@echo "Starting Single-Node Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh single_node "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)"

run_perf_hybrid: $(EXECUTABLE_HYBRID) $(SCRIPT_DIR)/run_benchmarks.sh
	@echo "Starting Hybrid Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh hybrid "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)" "$(P_VALUES_LIST)"

run_all_tests:
	$(MAKE) test_correctness_single_node N=10000 R_PAYLOAD=8 T_THREADS=2
	$(MAKE) test_correctness_hybrid N=10000 R_PAYLOAD=8 T_THREADS=2 P_PROCS=2
	$(MAKE) run_perf_single_node N_VALUES_LIST="100000" R_VALUES_LIST="64" T_VALUES_LIST="1 2"
	$(MAKE) run_perf_hybrid N_VALUES_LIST="100000" R_VALUES_LIST="64" T_VALUES_LIST="1" P_VALUES_LIST="1 2"

plot_results: $(SCRIPT_DIR)/plot_results.py
	@echo "Generating plots from benchmark results..."
	if [ -f results_single_node.csv ] && [ -f results_hybrid.csv ]; then \
		python3 $(SCRIPT_DIR)/plot_results.py \
			--input_ff results_single_node.csv \
			--input_hybrid results_hybrid.csv \
			--output_dir $(BUILD_DIR)/plots; \
	else \
		echo "Warning: One or both result CSV files not found. Skipping plot generation."; \
	fi
data:
	@echo "Data generation is handled on-demand by specific test targets."
