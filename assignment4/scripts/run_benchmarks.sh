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
NUM_RUNS=5
BASE_SEED=42

RESULTS_CSV="results_${MODE}.csv"

# --- Helper Functions ---
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

std_dev() {
    local data_points="$1"
    local count=$(echo "$data_points" | wc -w)
    if [ "$count" -lt 2 ]; then
        echo "0"
        return
    fi
    local m=$(mean "$data_points")
    local sum_sq_diff=0
    for val in $data_points; do
        local diff=$(echo "$val - $m" | bc -l)
        sum_sq_diff=$(echo "$sum_sq_diff + ($diff * $diff)" | bc -l)
    done
    # For sample standard deviation, divide by (count - 1). For population, divide by count.
    # Using (count - 1) as it's common for experimental runs.
    echo "sqrt($sum_sq_diff / ($count - 1))" | bc -l
}


# --- Script Start ---
echo "Starting Performance Benchmarks for ${MODE} mode..."
echo "Results will be saved to ${RESULTS_CSV}"
echo "Number of runs per configuration: ${NUM_RUNS}"

mkdir -p "$DATA_DIR"

if [ "$MODE" == "single_node" ]; then
    echo "N,R,T,mean_time_sec,std_dev_time_sec,min_time_sec,max_time_sec" > "$RESULTS_CSV"
else
    echo "N,R,T,P,mean_time_sec,std_dev_time_sec,min_time_sec,max_time_sec" > "$RESULTS_CSV"
fi

read -r -a N_VALUES <<< "$N_VALUES_LIST_STR"
read -r -a R_VALUES <<< "$R_VALUES_LIST_STR"
read -r -a T_VALUES <<< "$T_VALUES_LIST_STR"
if [ "$MODE" == "hybrid" ]; then
    read -r -a P_VALUES <<< "$P_VALUES_LIST_STR"
fi

current_seed=$BASE_SEED

for N_VAL in "${N_VALUES[@]}"; do
    for R_VAL in "${R_VALUES[@]}"; do

        DATA_FILE="${DATA_DIR}/perf_N${N_VAL}_R${R_VAL}_Seed${current_seed}.dat"
        echo "Generating data for N=${N_VAL}, R=${R_VAL} (Seed: ${current_seed}) -> ${DATA_FILE}"
        python3 "${GENERATOR_SCRIPT}" --size "${N_VAL}" --payload "${R_VAL}" --output "${DATA_FILE}" --seed "${current_seed}" --distribution "random"
        if [ $? -ne 0 ]; then
            echo "ERROR: Data generation failed for N=${N_VAL}, R=${R_VAL}. Skipping this combination."
            continue
        fi
        current_seed=$((current_seed + 1))

        for T_VAL in "${T_VALUES[@]}"; do
            if [ "$MODE" == "single_node" ]; then
                P_ITER_LIST=(0)
            else
                P_ITER_LIST=("${P_VALUES[@]}")
            fi

            for P_VAL in "${P_ITER_LIST[@]}"; do
                TIMES_STR=""

                CMD_BASE="${EXECUTABLE_PATH} -s ${N_VAL} -r ${R_VAL} -t ${T_VAL} --input ${DATA_FILE} --perf_mode"
                if [ "$MODE" == "hybrid" ]; then
                    CMD_RUN="mpirun -np ${P_VAL} --oversubscribe ${CMD_BASE}"
                    echo "Benchmarking: N=${N_VAL}, R=${R_VAL}, T_FF=${T_VAL}, P_MPI=${P_VAL}"
                else
                    CMD_RUN="${CMD_BASE}"
                    echo "Benchmarking: N=${N_VAL}, R=${R_VAL}, T_FF=${T_VAL}"
                fi

                # Initialize min_time to a very large number, max_time to a very small number (or first run's time)
                min_time=""
                max_time=""

                for i in $(seq 1 $NUM_RUNS); do
                    echo -n "  Run $i/$NUM_RUNS: "

                    run_output=$($CMD_RUN)
                    if [ $? -ne 0 ]; then
                        echo "ERROR during execution. Output: $run_output"
                        TIMES_STR=""
                        break
                    fi

                    exec_time=$(echo "$run_output" | awk -F',' '{print $NF}')

                    echo "${exec_time}s"
                    TIMES_STR="${TIMES_STR}${exec_time} "

                    if [ -z "$min_time" ] || (( $(echo "$exec_time < $min_time" |bc -l) )); then min_time=$exec_time; fi
                    if [ -z "$max_time" ] || (( $(echo "$exec_time > $max_time" |bc -l) )); then max_time=$exec_time; fi
                done

                if [ -z "$TIMES_STR" ]; then
                    echo "  Skipping results for this configuration due to errors."
                    continue
                fi

                mean_val=$(mean "$TIMES_STR")
                std_dev_val=$(std_dev "$TIMES_STR")

                if [ "$MODE" == "single_node" ]; then
                    echo "${N_VAL},${R_VAL},${T_VAL},${mean_val},${std_dev_val},${min_time},${max_time}" >> "$RESULTS_CSV"
                else
                    echo "${N_VAL},${R_VAL},${T_VAL},${P_VAL},${mean_val},${std_dev_val},${min_time},${max_time}" >> "$RESULTS_CSV"
                fi
                echo "  Mean: ${mean_val}s, StdDev: ${std_dev_val}s, Min: ${min_time}s, Max: ${max_time}s"
                echo ""
            done
        done
    done
done

echo "Benchmark suite finished. Results are in ${RESULTS_CSV}"
