#!/bin/bash

# run_benchmarks.sh
# Usage:
# ./run_benchmarks.sh <mode> "<N_list>" "<R_list>" "<T_list>" [ "<P_list>" ] <executable_path> <data_dir_path> <generator_script_path>
# mode: 'single_node' or 'hybrid'

# --- Parameter Parsing ---
if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 <mode> \"<N_list>\" \"<R_list>\" \"<T_list>\" [ \"<P_list>\" ] <executable_path> <data_dir_path> <generator_script_path>"
    echo "Example (single_node): $0 single_node \"1M 10M\" \"64 256\" \"1 2 4\" ./build/bin/sort_ff ./data ./scripts/generate_data.py"
    echo "Example (hybrid):      $0 hybrid \"1M 10M\" \"64 256\" \"1 2 4\" \"2 4\" ./build/bin/sort_hybrid ./data ./scripts/generate_data.py"
    exit 1
fi

MODE=$1
N_VALUES_LIST_STR=$2
R_VALUES_LIST_STR=$3
T_VALUES_LIST_STR=$4

P_VALUES_LIST_STR=""
EXECUTABLE_PATH=""
DATA_DIR=""
GENERATOR_SCRIPT=""

if [ "$MODE" == "single_node" ]; then
    if [ "$#" -ne 7 ]; then
        echo "Error: Incorrect number of arguments for single_node mode."
        exit 1
    fi
    EXECUTABLE_PATH=$5
    DATA_DIR=$6
    GENERATOR_SCRIPT=$7
elif [ "$MODE" == "hybrid" ]; then
    if [ "$#" -ne 8 ]; then
        echo "Error: Incorrect number of arguments for hybrid mode."
        exit 1
    fi
    P_VALUES_LIST_STR=$5
    EXECUTABLE_PATH=$6
    DATA_DIR=$7
    GENERATOR_SCRIPT=$8
else
    echo "Error: Invalid mode '$MODE'. Must be 'single_node' or 'hybrid'."
    exit 1
fi

# --- Configuration ---
NUM_RUNS=5 # Number of runs for each configuration to calculate mean and std_dev
BASE_SEED=42 # Base seed for data generation, can be incremented for different N,R pairs

# Output CSV file
RESULTS_CSV="results_${MODE}.csv"

# --- Helper Functions ---
# Function to calculate mean of a list of numbers
mean() {
    local sum=0
    local count=0
    for val in $1; do
        sum=$(echo "$sum + $val" | bc -l)
        count=$((count + 1))
    done
    if [ "$count" -gt 0 ]; then
        echo "$sum / $count" | bc -l
    else
        echo "0"
    fi
}

# Function to calculate standard deviation of a list of numbers
std_dev() {
    local data_points="$1"
    local count=$(echo "$data_points" | wc -w)
    if [ "$count" -lt 2 ]; then # Std dev not meaningful for < 2 points
        echo "0"
        return
    fi
    local m=$(mean "$data_points")
    local sum_sq_diff=0
    for val in $data_points; do
        local diff=$(echo "$val - $m" | bc -l)
        sum_sq_diff=$(echo "$sum_sq_diff + ($diff * $diff)" | bc -l)
    done
    echo "sqrt($sum_sq_diff / ($count - 1))" | bc -l
}


# --- Script Start ---
echo "Starting Performance Benchmarks for ${MODE} mode..."
echo "Results will be saved to ${RESULTS_CSV}"
echo "Number of runs per configuration: ${NUM_RUNS}"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Write CSV Header
if [ "$MODE" == "single_node" ]; then
    echo "N,R,T,mean_time_sec,std_dev_time_sec,min_time_sec,max_time_sec" > "$RESULTS_CSV"
else # hybrid
    echo "N,R,T,P,mean_time_sec,std_dev_time_sec,min_time_sec,max_time_sec" > "$RESULTS_CSV"
fi

# Convert string lists to arrays
read -r -a N_VALUES <<< "$N_VALUES_LIST_STR"
read -r -a R_VALUES <<< "$R_VALUES_LIST_STR"
read -r -a T_VALUES <<< "$T_VALUES_LIST_STR"
if [ "$MODE" == "hybrid" ]; then
    read -r -a P_VALUES <<< "$P_VALUES_LIST_STR"
fi

# --- Main Benchmark Loop ---
current_seed=$BASE_SEED

for N_VAL in "${N_VALUES[@]}"; do
    for R_VAL in "${R_VALUES[@]}"; do

        # Generate data ONCE for this N, R combination (using current_seed)
        # This ensures all thread/process counts for this N,R use the same input data.
        DATA_FILE="${DATA_DIR}/perf_N${N_VAL}_R${R_VAL}_Seed${current_seed}.dat"
        echo "Generating data for N=${N_VAL}, R=${R_VAL} (Seed: ${current_seed}) -> ${DATA_FILE}"
        python3 "${GENERATOR_SCRIPT}" --size "${N_VAL}" --payload "${R_VAL}" --output "${DATA_FILE}" --seed "${current_seed}" --distribution "random"
        if [ $? -ne 0 ]; then
            echo "ERROR: Data generation failed for N=${N_VAL}, R=${R_VAL}. Skipping this combination."
            continue
        fi
        current_seed=$((current_seed + 1)) # Use a different seed for the next N,R pair

        for T_VAL in "${T_VALUES[@]}"; do
            if [ "$MODE" == "single_node" ]; then
                P_ITER_LIST=(0) # Dummy list for single_node to fit loop structure
            else
                P_ITER_LIST=("${P_VALUES[@]}")
            fi

            for P_VAL in "${P_ITER_LIST[@]}"; do # Loop once for single_node, P_VAL will be 0
                TIMES_STR="" # String to hold times for this config

                CMD_BASE="${EXECUTABLE_PATH} -s ${N_VAL} -r ${R_VAL} -t ${T_VAL} --input ${DATA_FILE} --perf_mode"
                if [ "$MODE" == "hybrid" ]; then
                    CMD_RUN="mpirun -np ${P_VAL} --oversubscribe ${CMD_BASE}"
                    echo "Benchmarking: N=${N_VAL}, R=${R_VAL}, T_FF=${T_VAL}, P_MPI=${P_VAL}"
                else
                    CMD_RUN="${CMD_BASE}"
                    echo "Benchmarking: N=${N_VAL}, R=${R_VAL}, T_FF=${T_VAL}"
                fi

                min_time="Infinity"
                max_time="-Infinity"

                for i in $(seq 1 $NUM_RUNS); do
                    echo -n "  Run $i/$NUM_RUNS: "
                    # Execute the command and capture output (expected CSV: N,R,T,time or N,R,T,P,time)
                    # The --perf_mode output from main_ff.cpp is "N,R,T,time_sec"
                    # The --perf_mode output from main_hybrid.cpp should be "N,R,T,P,time_sec"

                    run_output=$($CMD_RUN)
                    if [ $? -ne 0 ]; then
                        echo "ERROR during execution. Output: $run_output"
                        # Decide if you want to skip this config or exit entirely
                        TIMES_STR="" # Invalidate times for this config
                        break
                    fi

                    # Extract time (last field of CSV output)
                    exec_time=$(echo "$run_output" | awk -F',' '{print $NF}')

                    echo "${exec_time}s"
                    TIMES_STR="${TIMES_STR}${exec_time} "

                    # Update min/max
                    if (( $(echo "$exec_time < $min_time" |bc -l) )); then min_time=$exec_time; fi
                    if (( $(echo "$exec_time > $max_time" |bc -l) )); then max_time=$exec_time; fi
                done

                if [ -z "$TIMES_STR" ]; then # If there was an error in runs
                    echo "  Skipping results for this configuration due to errors."
                    continue
                fi

                mean_val=$(mean "$TIMES_STR")
                std_dev_val=$(std_dev "$TIMES_STR")

                if [ "$MODE" == "single_node" ]; then
                    echo "${N_VAL},${R_VAL},${T_VAL},${mean_val},${std_dev_val},${min_time},${max_time}" >> "$RESULTS_CSV"
                else # hybrid
                    echo "${N_VAL},${R_VAL},${T_VAL},${P_VAL},${mean_val},${std_dev_val},${min_time},${max_time}" >> "$RESULTS_CSV"
                fi
                echo "  Mean: ${mean_val}s, StdDev: ${std_dev_val}s, Min: ${min_time}s, Max: ${max_time}s"
                echo "" # Newline for readability
            done # End P_VAL loop
        done # End T_VAL loop
    done # End R_VAL loop
done # End N_VAL loop

echo "Benchmark suite finished. Results are in ${RESULTS_CSV}"
