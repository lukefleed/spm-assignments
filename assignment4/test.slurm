#!/bin/bash
#SBATCH --job-name=hybrid_sort_test_4nodes  # Nome del job più descrittivo
#SBATCH --output=hybrid_sort_test_4nodes_%j.out # File di output standard
#SBATCH --error=hybrid_sort_test_4nodes_%j.err  # File di errore standard
#SBATCH --nodes=4                             # RICHIESTA: 4 NODI! (Modifica qui per 1, 2, 4, 8 nodi)
#SBATCH --ntasks-per-node=1                   # Esegui 1 processo MPI per nodo
#SBATCH --cpus-per-task=4                     # Ogni processo MPI userà 4 CPU (per FastFlow)
#SBATCH --time=00:30:00                       # Tempo massimo di esecuzione (HH:MM:SS)

cd $SLURM_SUBMIT_DIR

echo "Inizio benchmark ibrido (4 nodi)..."

# Parametri per il benchmark
export N_VALUES_LIST="1M 5M 10M"
export R_VALUES_LIST="64"
export T_VALUES_LIST="1 2 4"

# --- DEBUG: Stampa i valori delle variabili Slurm ---
echo "DEBUG: SLURM_JOB_ID          = ${SLURM_JOB_ID}"
echo "DEBUG: SLURM_NNODES          = ${SLURM_NNODES}"
echo "DEBUG: SLURM_NTASKS_PER_NODE = ${SLURM_NTASKS_PER_NODE}"
echo "DEBUG: SLURM_CPUS_PER_TASK   = ${SLURM_CPUS_PER_TASK}"
echo "DEBUG: SLURM_NTASKS (Total)  = ${SLURM_NTASKS}"
echo "------------------------------------------"
# --- FINE DEBUG ---

# --- DEBUG: Prepara gli argomenti per run_benchmarks.sh in un array ---
ARGS=(
    "hybrid"
    "${N_VALUES_LIST}"
    "${R_VALUES_LIST}"
    "${T_VALUES_LIST}"
    "${SLURM_NTASKS}"
    "./build/bin/sort_hybrid"
    "./data"
    "./scripts/generate_data.py"
)

echo "DEBUG: Arguments to run_benchmarks.sh:"
for i in "${!ARGS[@]}"; do
    echo "DEBUG:   Arg $((i+1)): '${ARGS[i]}'"
done
echo "DEBUG: Total arguments: ${#ARGS[@]}"
echo "------------------------------------------"
# --- FINE DEBUG ---


# Esegui lo script di benchmark
bash ./scripts/run_benchmarks.sh "${ARGS[@]}"

echo "Benchmark completato."
