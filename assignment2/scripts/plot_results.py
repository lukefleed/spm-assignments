#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt # Pyplot interface
import matplotlib.ticker as mticker
import os
import sys
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_FILE = RESULTS_DIR / "performance_data.csv"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def plot_speedup_pyplot(df, baseline_time_ms, output_filename="speedup_pyplot_plot.png"):
    """
    Generates and saves a speedup plot using the matplotlib.pyplot interface.

    Args:
        df (pd.DataFrame): DataFrame containing parallel execution results
                           (must include 'Scheduler', 'Threads', 'ChunkSize', 'MedianTimeMs').
        baseline_time_ms (float): The sequential execution time (T1) in milliseconds.
        output_filename (str): The name for the output plot file.
    """
    if baseline_time_ms <= 0:
        print("Error: Baseline sequential time is invalid.", file=sys.stderr)
        return

    # Calculate Speedup T(1) / T(P)
    df['Speedup'] = baseline_time_ms / df['MedianTimeMs']
    df.loc[df['MedianTimeMs'] <= 0, 'Speedup'] = 0

    # --- Plotting using pyplot ---
    plt.figure(figsize=(10, 6)) # Create a new figure

    schedulers = df['Scheduler'].unique()
    chunk_sizes = df['ChunkSize'].unique()
    colors = plt.cm.viridis(range(len(schedulers) * len(chunk_sizes)))
    color_idx = 0
    max_threads = 0

    # Plot results for each scheduler and chunk size combination
    for scheduler in sorted(schedulers):
        for chunk_size in sorted(chunk_sizes):
            subset = df[(df['Scheduler'] == scheduler) & (df['ChunkSize'] == chunk_size)].sort_values(by='Threads')
            if not subset.empty:
                label = f"{scheduler} (Chunk={chunk_size})"
                plt.plot(subset['Threads'], subset['Speedup'], # Use plt.plot
                         marker='o', linestyle='-',
                         label=label, color=colors[color_idx % len(colors)])
                color_idx += 1
                max_threads = max(max_threads, subset['Threads'].max())

    # Plot Ideal Speedup line
    if max_threads > 0:
        ideal_threads = sorted(df['Threads'].unique())
        if 1 not in ideal_threads:
            ideal_threads.insert(0, 1)
        ideal_threads = [t for t in ideal_threads if t <= max_threads]
        plt.plot(ideal_threads, ideal_threads, # Use plt.plot
                 label='Ideal Speedup', linestyle='--', color='black', alpha=0.7)

    # --- Plot Formatting using pyplot ---
    plt.title('Parallel Collatz Performance Speedup (pyplot)') # Use plt.title
    plt.xlabel('Number of Threads')                        # Use plt.xlabel
    plt.ylabel('Speedup (T_sequential / T_parallel)')    # Use plt.ylabel
    plt.legend(title="Configuration", loc='best')           # Use plt.legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Use plt.grid

    # Set x-axis ticks
    thread_ticks = sorted(df['Threads'].unique())
    if thread_ticks:
        plt.xticks(thread_ticks) # Use plt.xticks
        # If you need specific formatting (rarely needed for integers), use plt.gca()
        # plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Set y-axis limit
    max_speedup = df['Speedup'].max()
    if max_threads > 0:
        plt.ylim(bottom=0, top=max(max_speedup, max_threads) * 1.1) # Use plt.ylim
    else:
        plt.ylim(bottom=0)

    # Save the plot
    output_path = PLOTS_DIR / output_filename
    try:
        plt.tight_layout() # Adjust layout
        plt.savefig(output_path, dpi=300) # Use plt.savefig
        print(f"Speedup plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)

    # Optional: Display the plot
    # plt.show()
    plt.close() # Close the figure after saving to free memory


# --- Main Execution Logic ---
# (This part remains unchanged from the previous version)
def main():
    print(f"Reading performance data from: {CSV_FILE}")

    if not CSV_FILE.is_file():
        print(f"Error: CSV file not found at {CSV_FILE}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    required_columns = ['Scheduler', 'Threads', 'ChunkSize', 'MedianTimeMs']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV missing required columns. Found: {list(df.columns)}. Expected: {required_columns}", file=sys.stderr)
        sys.exit(1)

    df = df[df['MedianTimeMs'] != 'ERROR']
    try:
        df['MedianTimeMs'] = pd.to_numeric(df['MedianTimeMs'])
    except ValueError as e:
         print(f"Error converting 'MedianTimeMs' to numeric: {e}. Check CSV format.", file=sys.stderr)
         sys.exit(1)

    df = df[df['MedianTimeMs'] > 0]

    sequential_df = df[df['Scheduler'] == 'Sequential']
    if sequential_df.empty:
        print("Error: No 'Sequential' entry found in CSV for baseline time.", file=sys.stderr)
        sys.exit(1)
    if len(sequential_df) > 1:
         print("Warning: Multiple 'Sequential' entries found. Using the first one.", file=sys.stderr)
    baseline_time_ms = sequential_df['MedianTimeMs'].iloc[0]
    print(f"Using Sequential baseline time (T1): {baseline_time_ms:.4f} ms")

    parallel_df = df[df['Scheduler'] != 'Sequential'].copy()
    if parallel_df.empty:
        print("Warning: No parallel execution data found in the CSV to plot.", file=sys.stderr)
        sys.exit(0)

    # Call the pyplot-based plotting function
    plot_speedup_pyplot(parallel_df, baseline_time_ms, output_filename="collatz_speedup_pyplot.png")


if __name__ == "__main__":
    main()
