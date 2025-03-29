#!/usr/bin/env python3

import pandas as pd
import os
import sys
from pathlib import Path
import plotly.graph_objects as go  # New import for Plotly

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_FILE = RESULTS_DIR / "performance_data.csv"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def plot_speedup_plotly(df, baseline_time_ms, output_filename="collatz_speedup_plotly.pdf"):
    """
    Generates and saves a speedup plot using Plotly.

    Args:
        df (pd.DataFrame): DataFrame with parallel execution results
                           (must include 'Scheduler', 'Threads', 'ChunkSize', 'MedianTimeMs').
        baseline_time_ms (float): The sequential execution time (T1) in milliseconds.
        output_filename (str): The name for the output plot file.
    """
    if baseline_time_ms <= 0:
        print("Error: Baseline sequential time is invalid.", file=sys.stderr)
        return

    # Calculate Speedup T(1) / T(P)
    df = df.copy()
    df['Speedup'] = baseline_time_ms / df['MedianTimeMs']
    df.loc[df['MedianTimeMs'] <= 0, 'Speedup'] = 0

    schedulers = sorted(df['Scheduler'].unique())
    chunk_sizes = sorted(df['ChunkSize'].unique())
    max_threads = df['Threads'].max() if not df['Threads'].empty else 0

    fig = go.Figure()

    # Plot each scheduler and chunk size combination
    for scheduler in schedulers:
        for chunk_size in chunk_sizes:
            subset = df[(df['Scheduler'] == scheduler) & (df['ChunkSize'] == chunk_size)].sort_values('Threads')
            if not subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=subset['Threads'],
                        y=subset['Speedup'],
                        mode='lines+markers',
                        name=f"{scheduler} (Chunk={chunk_size})"
                    )
                )

    # Plot Ideal Speedup line
    if max_threads:
        ideal_threads = sorted(df['Threads'].unique())
        if 1 not in ideal_threads:
            ideal_threads.insert(0, 1)
        ideal_threads = [t for t in ideal_threads if t <= max_threads]
        fig.add_trace(
            go.Scatter(
                x=ideal_threads,
                y=ideal_threads,
                mode='lines',
                line=dict(dash='dash', color='black'),
                name='Ideal Speedup'
            )
        )

    # Layout formatting with legend in the top left and a smaller font size
    fig.update_layout(
        title="Parallel Collatz Performance Speedup (Plotly)",
        xaxis_title="Number of Threads",
        yaxis_title="Speedup (T_sequential / T_parallel)",
        legend=dict(
            title="Configuration",
            x=0,  # left
            y=1,  # top
            font=dict(size=10)  # smaller font size
        ),
        xaxis=dict(tickmode='array', tickvals=sorted(df['Threads'].unique())),
        yaxis=dict(range=[0, max(max(df['Speedup']), max_threads) * 1.1])
    )

    output_path = PLOTS_DIR / output_filename
    try:
        # Save as a static PDF image (requires kaleido)
        fig.write_image(str(output_path), scale=2)
        print(f"Speedup plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot image: {e}", file=sys.stderr)


# --- Main Execution Logic ---
def main():
    print(f"Reading performance data from: {CSV_FILE}")

    if not CSV_FILE.is_file():
        print(f"Error: CSV file not found at {CSV_FILE}", file=sys.stderr)
        sys.exit(1)

    try:
        # Add header=None and specify column names
        df = pd.read_csv(CSV_FILE, header=None,
                         names=['Scheduler', 'Threads', 'ChunkSize', 'MedianTimeMs'])
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

    # Call the Plotly-based plotting function and output a PDF file
    plot_speedup_plotly(parallel_df, baseline_time_ms, output_filename="collatz_speedup_plotly.pdf")


if __name__ == "__main__":
    main()
