# Compiler and flags
CXX = g++
MPICXX = mpicxx # MPI C++ compiler
CXX_STD = -std=c++17
OPTIM_FLAGS = -O3 -march=native # Use -O2 or -O0 for easier debugging if needed
DEBUG_FLAGS = -g -Wall -Wextra -Wpedantic
COMMON_FLAGS = $(CXX_STD) $(DEBUG_FLAGS)

# FastFlow specific paths (ensure this path is correct for your setup)
# If FastFlow is installed system-wide or via a module, FF_HOME might not be needed,
# and FF_INC might be just -I/path/to/fastflow/include or handled by environment.
# For this project, we assume it's in a local directory.
FF_HOME ?= ./fastflow # Allows overriding from command line if FF is elsewhere
FF_INC = -I$(FF_HOME)

# MPI specific paths (usually handled by mpicxx wrapper)
MPI_INC =
MPI_LIB =

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
SCRIPT_DIR = scripts
DATA_DIR = data # For generated test data

# Common object files (shared between FF and Hybrid, compiled with CXX as they don't use MPI APIs directly)
COMMON_NON_MPI_SRCS = $(SRC_DIR)/utils.cpp $(SRC_DIR)/performance_timer.cpp $(SRC_DIR)/mergesort_common.cpp
COMMON_NON_MPI_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_NON_MPI_SRCS))

# Single-node (FastFlow) specific files
FF_MODULE_SRCS = $(SRC_DIR)/mergesort_ff.cpp # FF-specific sorting logic
FF_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_MODULE_SRCS))
FF_MAIN_SRC = $(SRC_DIR)/main_ff.cpp
FF_MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FF_MAIN_SRC))
EXECUTABLE_FF = $(BIN_DIR)/sort_ff

# Hybrid (MPI + FastFlow) specific files
HYBRID_MODULE_SRCS = $(SRC_DIR)/mergesort_hybrid.cpp # Hybrid-specific logic (e.g., k-way merge)
HYBRID_MODULE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_MODULE_SRCS))
HYBRID_MAIN_SRC = $(SRC_DIR)/main_hybrid.cpp
HYBRID_MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(HYBRID_MAIN_SRC))
EXECUTABLE_HYBRID = $(BIN_DIR)/sort_hybrid

.PHONY: all clean directories test_correctness_single_node test_correctness_hybrid run_perf_single_node run_perf_hybrid run_all_tests plot_results data test_correctness_suite_ff test_one_ff_config test_one_hybrid_config test_correctness_suite_hybrid

all: directories $(EXECUTABLE_FF) $(EXECUTABLE_HYBRID)

directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(DATA_DIR) $(BUILD_DIR)/plots

# --- Compilation Rules ---

# Generic rule for .o files from .cpp files in SRC_DIR that DON'T use MPI directly
# (utils.o, performance_timer.o, mergesort_common.o, mergesort_ff.o)
# Note: mergesort_ff.o is compiled with CXX as it's the FF library part.
$(OBJ_DIR)/utils.o $(OBJ_DIR)/performance_timer.o $(OBJ_DIR)/mergesort_common.o $(OBJ_DIR)/mergesort_ff.o: $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

# Rule for FF main application (non-MPI)
$(OBJ_DIR)/main_ff.o: $(SRC_DIR)/main_ff.cpp
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) -c $< -o $@

# Rule for Hybrid main application (USES MPI)
$(OBJ_DIR)/main_hybrid.o: $(SRC_DIR)/main_hybrid.cpp
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@

# Rule for Hybrid-specific modules (e.g., k-way merge)
# This is compiled with MPICXX in case it needs MPI types or simple MPI calls,
# though for a sequential k-way merge on root, CXX would also be fine if it takes plain pointers/counts.
# Using MPICXX is safer if there's any doubt or future MPI use within this module.
$(OBJ_DIR)/mergesort_hybrid.o: $(SRC_DIR)/mergesort_hybrid.cpp
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) -I$(INC_DIR) $(FF_INC) $(MPI_INC) -c $< -o $@


# --- Linking Executables ---
$(EXECUTABLE_FF): $(COMMON_NON_MPI_OBJS) $(FF_MODULE_OBJS) $(FF_MAIN_OBJ)
	$(CXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ -lpthread -o $@

# For HYBRID, we need:
# - COMMON_NON_MPI_OBJS (utils, timer, common sort helpers)
# - FF_MODULE_OBJS (mergesort_ff.o for local sort on each node)
# - HYBRID_MODULE_OBJS (mergesort_hybrid.o for k-way merge etc.)
# - HYBRID_MAIN_OBJ (main_hybrid.o)
$(EXECUTABLE_HYBRID): $(COMMON_NON_MPI_OBJS) $(FF_MODULE_OBJS) $(HYBRID_MODULE_OBJS) $(HYBRID_MAIN_OBJ)
	$(MPICXX) $(COMMON_FLAGS) $(OPTIM_FLAGS) $^ $(MPI_LIB) -lpthread -o $@


# --- Default execution parameters ---
N ?= 1M
R_PAYLOAD ?= 64
T_THREADS ?= 4 # FastFlow threads per MPI process for hybrid, or total for FF
P_PROCS ?= 2   # MPI processes for hybrid
DIST ?= random
SEED ?= 42

# --- Correctness Testing Targets ---
# (test_correctness_single_node, test_correctness_suite_ff, test_one_ff_config are from your existing file and should be fine)
# You can copy them here. I'll add the hybrid ones.

# Ad-hoc single test target for FF
test_correctness_single_node: $(EXECUTABLE_FF) $(SCRIPT_DIR)/generate_data.py
	@echo "Running Ad-hoc Correctness Test for Single-Node: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS), Dist=$(DIST), Seed=$(SEED)"
	$(eval TEST_DATA_FILE_ADHOC = $(DATA_DIR)/input_N$(N)_R$(R_PAYLOAD)_Dist$(DIST)_Seed$(SEED).dat)
	python3 $(SCRIPT_DIR)/generate_data.py --size "$(N)" --payload "$(R_PAYLOAD)" --output "$(TEST_DATA_FILE_ADHOC)" --seed "$(SEED)" --distribution "$(DIST)"
	./$(EXECUTABLE_FF) -s "$(N)" -r "$(R_PAYLOAD)" -t "$(T_THREADS)" --input "$(TEST_DATA_FILE_ADHOC)" --check_correctness --perf_mode
	@if [ $$? -eq 0 ]; then echo "    STATUS: PASSED (Ad-hoc FF)"; else echo "    STATUS: FAILED (Ad-hoc FF)"; exit 1; fi; echo ""

# Correctness test suite for FastFlow version
test_correctness_suite_ff: directories $(EXECUTABLE_FF) $(SCRIPT_DIR)/generate_data.py
	@echo "\n######################################################"
	@echo "# RUNNING FULL FF CORRECTNESS TEST SUITE             #"
	@echo "######################################################"
	$(MAKE) test_one_ff_config N=10K R_PAYLOAD=8   T_THREADS=2 DIST=random        SEED=101
	$(MAKE) test_one_ff_config N=10K R_PAYLOAD=256 T_THREADS=4 DIST=sorted_asc    SEED=102
	# ... (add more configurations from your original Makefile)
	$(MAKE) test_one_ff_config N=1M R_PAYLOAD=64 T_THREADS=4 DIST=random SEED=107
	@echo "######################################################"
	@echo "# FF CORRECTNESS TEST SUITE COMPLETED                #"
	@echo "######################################################\n"

test_one_ff_config:
	@echo "--- Testing FF: N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS), Dist=$(DIST), Seed=$(SEED) ---"
	$(eval TEST_DATA_FILE_NAME = input_N$(N)_R$(R_PAYLOAD)_Dist$(DIST)_Seed$(SEED).dat)
	python3 $(SCRIPT_DIR)/generate_data.py --size "$(N)" --payload "$(R_PAYLOAD)" --output "$(DATA_DIR)/$(TEST_DATA_FILE_NAME)" --seed "$(SEED)" --distribution "$(DIST)"
	./$(EXECUTABLE_FF) -s "$(N)" -r "$(R_PAYLOAD)" -t "$(T_THREADS)" --input "$(DATA_DIR)/$(TEST_DATA_FILE_NAME)" --check_correctness --perf_mode
	@if [ $$? -eq 0 ]; then echo "    STATUS: PASSED"; else echo "    STATUS: FAILED"; exit 1; fi
	@echo "--- Test FF N=$(N), R=$(R_PAYLOAD), T=$(T_THREADS), Dist=$(DIST), Seed=$(SEED) COMPLETED ---\n"

# Ad-hoc single test target for Hybrid
test_correctness_hybrid: $(EXECUTABLE_HYBRID) $(SCRIPT_DIR)/generate_data.py
	@echo "Running Ad-hoc Correctness Test for Hybrid: N=$(N), R=$(R_PAYLOAD), T_FF=$(T_THREADS), P_MPI=$(P_PROCS), Dist=$(DIST), Seed=$(SEED)"
	$(eval TEST_DATA_FILE_ADHOC_HYBRID = $(DATA_DIR)/input_N$(N)_R$(R_PAYLOAD)_Dist$(DIST)_Seed$(SEED)_hybrid.dat)
	python3 $(SCRIPT_DIR)/generate_data.py --size "$(N)" --payload "$(R_PAYLOAD)" --output "$(TEST_DATA_FILE_ADHOC_HYBRID)" --seed "$(SEED)" --distribution "$(DIST)"
	mpirun -np $(P_PROCS) --oversubscribe \
		./$(EXECUTABLE_HYBRID) -s "$(N)" -r "$(R_PAYLOAD)" -t "$(T_THREADS)" \
		--input "$(TEST_DATA_FILE_ADHOC_HYBRID)" --check_correctness --perf_mode
	@if [ $$? -eq 0 ]; then echo "    STATUS: PASSED (Ad-hoc Hybrid)"; else echo "    STATUS: FAILED (Ad-hoc Hybrid)"; exit 1; fi; echo ""

# Correctness test suite for Hybrid version
test_correctness_suite_hybrid: directories $(EXECUTABLE_HYBRID) $(SCRIPT_DIR)/generate_data.py
	@echo "\n######################################################"
	@echo "# RUNNING FULL HYBRID CORRECTNESS TEST SUITE         #"
	@echo "######################################################"
	$(MAKE) test_one_hybrid_config N=10K R_PAYLOAD=8   T_THREADS=1 P_PROCS=2 DIST=random        SEED=201
	$(MAKE) test_one_hybrid_config N=10K R_PAYLOAD=256 T_THREADS=2 P_PROCS=4 DIST=sorted_desc   SEED=202
	# ... (add more configurations as needed)
	$(MAKE) test_one_hybrid_config N=1M R_PAYLOAD=64 T_THREADS=2 P_PROCS=2 DIST=random SEED=203
	@echo "######################################################"
	@echo "# HYBRID CORRECTNESS TEST SUITE COMPLETED            #"
	@echo "######################################################\n"

test_one_hybrid_config:
	@echo "--- Testing Hybrid: N=$(N), R=$(R_PAYLOAD), T_FF=$(T_THREADS), P_MPI=$(P_PROCS), Dist=$(DIST), Seed=$(SEED) ---"
	$(eval TEST_DATA_FILE_NAME = input_N$(N)_R$(R_PAYLOAD)_Dist$(DIST)_Seed$(SEED)_hybrid.dat)
	python3 $(SCRIPT_DIR)/generate_data.py --size "$(N)" --payload "$(R_PAYLOAD)" --output "$(DATA_DIR)/$(TEST_DATA_FILE_NAME)" --seed "$(SEED)" --distribution "$(DIST)"
	mpirun -np $(P_PROCS) --oversubscribe \
		./$(EXECUTABLE_HYBRID) -s "$(N)" -r "$(R_PAYLOAD)" -t "$(T_THREADS)" \
		--input "$(DATA_DIR)/$(TEST_DATA_FILE_NAME)" --check_correctness --perf_mode
	@if [ $$? -eq 0 ]; then echo "    STATUS: PASSED"; else echo "    STATUS: FAILED"; exit 1; fi
	@echo "--- Test Hybrid N=$(N), R=$(R_PAYLOAD), T_FF=$(T_THREADS), P_MPI=$(P_PROCS), Dist=$(DIST), Seed=$(SEED) COMPLETED ---\n"


# --- Performance Testing Targets ---
N_VALUES_LIST ?= "10M 50M"
R_VALUES_LIST ?= "64 256"
T_VALUES_LIST ?= "1 2 4 8" # FF Threads per node for hybrid, or total for FF
P_VALUES_LIST ?= "1 2 4"   # MPI Processes (nodes) for hybrid

run_perf_single_node: $(EXECUTABLE_FF) $(SCRIPT_DIR)/run_benchmarks.sh $(SCRIPT_DIR)/generate_data.py
	@echo "Starting Single-Node Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh single_node "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)" ./$(EXECUTABLE_FF) $(DATA_DIR) $(SCRIPT_DIR)/generate_data.py

run_perf_hybrid: $(EXECUTABLE_HYBRID) $(SCRIPT_DIR)/run_benchmarks.sh $(SCRIPT_DIR)/generate_data.py
	@echo "Starting Hybrid Performance Benchmark Suite..."
	bash $(SCRIPT_DIR)/run_benchmarks.sh hybrid "$(N_VALUES_LIST)" "$(R_VALUES_LIST)" "$(T_VALUES_LIST)" "$(P_VALUES_LIST)" ./$(EXECUTABLE_HYBRID) $(DATA_DIR) $(SCRIPT_DIR)/generate_data.py

# --- Other Targets ---
run_all_tests: # Runs quick correctness tests
	$(MAKE) test_correctness_suite_ff N=10K R_PAYLOAD=64 T_THREADS=2 SEED=1
	$(MAKE) test_correctness_suite_hybrid N=10K R_PAYLOAD=64 T_THREADS=1 P_PROCS=2 SEED=2

plot_results: $(SCRIPT_DIR)/plot_results.py
	@echo "Generating plots from benchmark results..."
	if [ -f results_single_node.csv ] || [ -f results_hybrid.csv ]; then \
		python3 $(SCRIPT_DIR)/plot_results.py \
			results_single_node.csv \
			results_hybrid.csv \
			$(BUILD_DIR)/plots; \
	else \
		echo "Warning: One or both result CSV files not found. Run benchmarks first. Skipping plot generation."; \
	fi

data: # Placeholder, data generation is part of test/benchmark targets
	@echo "Data generation is handled on-demand by specific test targets or benchmark scripts."

clean:
	rm -rf $(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR)
	rm -f $(DATA_DIR)/input_N*_R*_Dist*_Seed*.dat
	rm -f $(DATA_DIR)/perf_N*_R*_Seed*.dat
	rm -f results_single_node.csv results_hybrid.csv
	@echo "Cleaned build artifacts, object files, executables, generated data, and result CSVs."
