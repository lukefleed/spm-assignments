import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# --- Patch NumPy bool deprecation ---
if not hasattr(np, 'bool'):
    np.bool = np.bool_

# --- Constants & Configuration ---
DEFAULT_CSV_PATH = "../results/performance_results_sencha.csv"
DEFAULT_PLOT_DIR = "../results/plots_sencha"
DEFAULT_FIXED_CHUNK = 64
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
HEATMAP_WIDTH = 800
HEATMAP_HEIGHT = 800
MAX_FILTER_CHUNK_SIZE = 1024 # Ignore excessively large chunks in chunk plots

SCHEDULER_COLOR_MAP = {
    "Sequential": "black",
    "Static Block": px.colors.qualitative.Plotly[0],
    "Static Cyclic": px.colors.qualitative.Plotly[1],
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic": px.colors.qualitative.Plotly[3]
}

CHUNK_SCHEDULER_COLOR_MAP = {
    "Static Block": px.colors.qualitative.Plotly[0],
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic": px.colors.qualitative.Plotly[3]
}

SCHEDULERS_WITH_CHUNK = ["Static Block", "Static Block-Cyclic", "Dynamic"]
SCHEDULERS_NO_CHUNK = ["Sequential", "Static Cyclic"]

# --- Helper Functions ---

def _prepare_output_dir(base_plot_dir: Path, sub_directory: str) -> Path:
    """Creates the subdirectory within the base plot directory and returns its path."""
    output_dir = base_plot_dir / sub_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _save_figure(fig: go.Figure, filepath: Path):
    """Saves the Plotly figure to a PDF file with error handling."""
    try:
        fig.write_image(filepath, format="pdf")
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  ERROR saving {filepath}: {e}. Ensure Kaleido is installed ('pip install kaleido').")

def _configure_xaxis_ticks(fig: go.Figure, unique_values: List[int], is_threads: bool):
    """Configures x-axis ticks for threads or chunk sizes."""
    if not unique_values: return
    max_val = max(unique_values)
    if len(unique_values) < 8:
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted(unique_values))
    elif is_threads:
        dtick = 2 if max_val <= 16 else 4
        fig.update_xaxes(type='linear', dtick=dtick)
    else: # Chunk sizes can be more spread out, linear might be better if many values
         fig.update_xaxes(type='linear') # Or keep category if preferred: type='category', categoryorder='array', categoryarray=sorted(unique_values)

def _add_amdahl_trace(fig: go.Figure, group_df: pd.DataFrame):
    """Calculates and adds Amdahl's Law trace to a speedup plot."""
    if 'NumThreads' not in group_df.columns or 'Speedup' not in group_df.columns:
        return

    unique_threads = sorted(group_df['NumThreads'].unique())
    if not unique_threads: return

    max_threads = max(unique_threads)
    if max_threads <= 1: return

    max_thread_data = group_df[group_df['NumThreads'] == max_threads]
    if max_thread_data.empty: return

    best_speedup = max_thread_data['Speedup'].max()

    if pd.notna(best_speedup) and np.isfinite(best_speedup) and best_speedup > 1:
        # S(n) = 1 / (s + (1-s)/n) => s = (n/S(n) - 1) / (n - 1)
        s = (max_threads / best_speedup - 1) / (max_threads - 1)
        s = max(0.01, min(0.99, s)) # Clamp s

        amdahl_x = np.linspace(1, max_threads, 100)
        amdahl_y = [1 / (s + (1 - s) / n) if n > 0 else 1 for n in amdahl_x]

        fig.add_trace(go.Scatter(
            x=amdahl_x, y=amdahl_y, mode='lines',
            line=dict(color='red', dash='dash', width=1.5),
            name=f"Amdahl's Law (s={s:.2f})", showlegend=True
        ))

def _filter_and_sort(df: pd.DataFrame, required_cols: List[str], sort_by: List[str],
                     filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Applies common filtering, NaN dropping, type conversion, and sorting."""
    df_filtered = df.dropna(subset=required_cols).copy()
    if 'NumThreads' in required_cols:
        df_filtered = df_filtered[df_filtered['NumThreads'] >= 1]
        df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)
    if 'ChunkSize' in required_cols:
        # Ensure ChunkSize is numeric before filtering, handle potential prior NaNs
        df_filtered = df_filtered[pd.to_numeric(df_filtered['ChunkSize'], errors='coerce').notna()]
        df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
        df_filtered = df_filtered[df_filtered['ChunkSize'] > 0]
        df_filtered = df_filtered[df_filtered['ChunkSize'] <= MAX_FILTER_CHUNK_SIZE]
    if 'ExecutionTimeMs' in required_cols:
        df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]
    if filters:
        for col, value in filters.items():
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[col] == value]

    df_filtered.sort_values(by=sort_by, inplace=True)
    return df_filtered


# --- Plotting Functions ---

def plot_speedup_vs_threads(df: pd.DataFrame, plot_dir: Path, fixed_chunk_size: int, file_suffix: str = ""):
    """Generates Speedup vs Number of Threads plots."""
    print(f"Plotting Speedup vs Threads (fixed chunk: {fixed_chunk_size})...")
    output_dir = _prepare_output_dir(plot_dir, "speedup_vs_threads")

    # Filter data specifically for this plot type
    df_plot = df[
        (pd.to_numeric(df['ChunkSize'], errors='coerce') == fixed_chunk_size) |
        (df['SchedulerName'].isin(SCHEDULERS_NO_CHUNK)) |
        (df['ChunkSize'].isna()) # Include rows where chunk size is irrelevant/NaN
    ].copy()

    df_plot = _filter_and_sort(df_plot, ['NumThreads', 'Speedup', 'WorkloadID', 'SchedulerName', 'WorkloadDescription'],
                               sort_by=['WorkloadID', 'SchedulerName', 'NumThreads'])

    for workload_id, group in df_plot.groupby('WorkloadID'):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"speedup_vs_threads_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group, x='NumThreads', y='Speedup', color='SchedulerName', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)'},
                      color_discrete_map=SCHEDULER_COLOR_MAP)

        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")
        _configure_xaxis_ticks(fig, sorted(group['NumThreads'].unique()), is_threads=True)
        _add_amdahl_trace(fig, group)

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

def plot_time_vs_threads(df: pd.DataFrame, plot_dir: Path, fixed_chunk_size: int, use_log_scale: bool, file_suffix: str = ""):
    """Generates Execution Time vs Number of Threads plots."""
    log_suffix = "_log" if use_log_scale else ""
    print(f"Plotting Execution Time vs Threads (fixed chunk: {fixed_chunk_size}, log_scale: {use_log_scale})...")
    output_dir = _prepare_output_dir(plot_dir, "time_vs_threads")

    # Filter data specifically for this plot type
    df_plot = df[
        (pd.to_numeric(df['ChunkSize'], errors='coerce') == fixed_chunk_size) |
        (df['SchedulerName'].isin(SCHEDULERS_NO_CHUNK)) |
        (df['ChunkSize'].isna())
    ].copy()

    df_plot = _filter_and_sort(df_plot, ['NumThreads', 'ExecutionTimeMs', 'WorkloadID', 'SchedulerName', 'WorkloadDescription'],
                               sort_by=['WorkloadID', 'SchedulerName', 'NumThreads'])

    for workload_id, group in df_plot.groupby('WorkloadID'):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
        title = f"Exec Time vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"time_vs_threads_W{workload_id}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group, x='NumThreads', y='ExecutionTimeMs', color='SchedulerName', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale, color_discrete_map=SCHEDULER_COLOR_MAP)

        _configure_xaxis_ticks(fig, sorted(group['NumThreads'].unique()), is_threads=True)
        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)


def plot_chunk_impact(df: pd.DataFrame, plot_dir: Path, metric: str, use_log_scale: bool = False, file_suffix: str = ""):
    """Generates plots for Speedup or Execution Time vs Chunk Size."""
    metric_label = 'Speedup' if metric == 'Speedup' else 'Execution Time (ms)'
    metric_col = 'Speedup' if metric == 'Speedup' else 'ExecutionTimeMs'
    log_suffix = "_log" if use_log_scale and metric == 'ExecutionTimeMs' else ""
    subdir = f"chunk_impact_{metric.lower()}"
    print(f"Plotting {metric_label} vs Chunk Size impact (log_scale: {use_log_scale})...")
    output_dir = _prepare_output_dir(plot_dir, subdir)

    df_plot = _filter_and_sort(df, ['ChunkSize', 'NumThreads', metric_col, 'WorkloadID', 'SchedulerName', 'WorkloadDescription'],
                               sort_by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'],
                               filters={'SchedulerName': SCHEDULERS_WITH_CHUNK, 'NumThreads': list(df[df['NumThreads'] > 1]['NumThreads'].unique())}) # Only parallel

    for (workload_id, num_threads), group in df_plot.groupby(['WorkloadID', 'NumThreads']):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        y_axis_label = metric_label + (' [Log Scale]' if use_log_scale and metric == 'ExecutionTimeMs' else '')
        title = f"{metric_label} vs Chunk Size - Workload: {workload_desc}<br>(Threads = {num_threads})"
        filename = f"chunk_{metric.lower()}_W{workload_id}_T{num_threads}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Use category for chunk size axis as they are discrete, potentially non-linear values
        group['ChunkSize_cat'] = group['ChunkSize'].astype(str)
        sorted_chunks_str = sorted(group['ChunkSize'].unique(), key=int) # Sort numerically but keep as strings for axis

        fig = px.line(group, x='ChunkSize_cat', y=metric_col, color='SchedulerName', markers=True, title=title,
                      labels={'ChunkSize_cat': 'Chunk Size', metric_col: y_axis_label},
                      log_y=use_log_scale if metric == 'ExecutionTimeMs' else False,
                      color_discrete_map=CHUNK_SCHEDULER_COLOR_MAP)

        # Ensure chunks are ordered correctly on the category axis
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=[str(c) for c in sorted_chunks_str])
        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)


def plot_scheduler_chunk_comparison(df: pd.DataFrame, plot_dir: Path, scheduler_name: str, file_suffix: str = ""):
    """Generates Speedup vs Threads plots for a single scheduler, comparing chunk sizes."""
    print(f"Plotting {scheduler_name} Speedup vs Threads for different chunk sizes...")
    output_dir = _prepare_output_dir(plot_dir, f"{scheduler_name.lower().replace(' ', '_')}_comparison")

    df_scheduler = _filter_and_sort(df, ['ChunkSize', 'NumThreads', 'Speedup', 'WorkloadID', 'SchedulerName', 'WorkloadDescription'],
                                    sort_by=['WorkloadID', 'ChunkSize', 'NumThreads'],
                                    filters={'SchedulerName': scheduler_name, 'NumThreads': list(df[df['NumThreads'] > 1]['NumThreads'].unique())})

    # Get unique chunks present for this scheduler
    chunk_sizes = sorted(df_scheduler['ChunkSize'].unique())
    if not chunk_sizes: return

    # Generate distinct colors for chunk sizes
    color_sequence = px.colors.sequential.Viridis
    n_chunks = len(chunk_sizes)
    color_map = {chunk: color_sequence[min(i * len(color_sequence) // n_chunks, len(color_sequence) - 1)]
                 for i, chunk in enumerate(chunk_sizes)}

    for workload_id, group in df_scheduler.groupby('WorkloadID'):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"{scheduler_name} Speedup vs Threads - Workload: {workload_desc}<br>(Comparison of different chunk sizes)"
        filename = f"{scheduler_name.lower().replace(' ', '_')}_chunks_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        group['ChunkSize_str'] = "Chunk=" + group['ChunkSize'].astype(str)

        fig = px.line(group, x='NumThreads', y='Speedup', color='ChunkSize_str', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)', 'ChunkSize_str': 'Chunk Size'},
                      color_discrete_map={f"Chunk={k}": v for k, v in color_map.items()})

        unique_threads = sorted(group['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")
        _add_amdahl_trace(fig, group) # Amdahl based on best across chunks for max threads

        fig.update_layout(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)


def plot_performance_heatmap(df: pd.DataFrame, plot_dir: Path, metric: str, pivot_index: str, pivot_columns: str, file_suffix: str = ""):
    """Generates heatmaps visualizing scheduler performance (Speedup or Time)."""
    metric_label = 'Speedup' if metric == 'Speedup' else 'Execution Time (ms)'
    metric_col = 'Speedup' if metric == 'Speedup' else 'ExecutionTimeMs'
    colorscale = 'Viridis' if metric == 'Speedup' else 'Viridis_r' # Higher is better for speedup, lower for time
    subdir = f"{metric.lower()}_heatmaps" # Adjust subdir based on usage if needed
    index_label = "Chunk Size" if pivot_index == 'ChunkSize' else "Number of Threads"
    columns_label = "Number of Threads" if pivot_columns == 'NumThreads' else "Chunk Size"

    print(f"Plotting {metric_label} heatmaps ({pivot_index} vs {pivot_columns})...")
    output_dir = _prepare_output_dir(plot_dir, subdir)

    df_heatmap = _filter_and_sort(df, ['ChunkSize', 'NumThreads', metric_col, 'WorkloadID', 'SchedulerName', 'WorkloadDescription'],
                                  sort_by=['WorkloadID', 'SchedulerName', pivot_index, pivot_columns],
                                  filters={'SchedulerName': SCHEDULERS_WITH_CHUNK, 'NumThreads': list(df[df['NumThreads'] > 1]['NumThreads'].unique())})

    for (workload_id, scheduler_name), group in df_heatmap.groupby(['WorkloadID', 'SchedulerName']):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]

        try:
            pivot_data = group.pivot_table(index=pivot_index, columns=pivot_columns, values=metric_col, aggfunc='mean')
        except Exception as e:
            print(f"  ERROR creating pivot table for W{workload_id}, {scheduler_name} ({metric}): {e}")
            continue

        if pivot_data.empty: continue

        # Sort indices/columns numerically for correct heatmap axis ordering
        pivot_data = pivot_data.sort_index(axis=0, key=lambda x: pd.to_numeric(x))
        pivot_data = pivot_data.sort_index(axis=1, key=lambda x: pd.to_numeric(x))

        title = f"{scheduler_name} {metric_label} Heatmap - {workload_desc}<br>({index_label} vs {columns_label})"
        filename = f"heatmap_{metric.lower()}_{scheduler_name.lower().replace(' ', '_')}_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.imshow(pivot_data,
                        labels=dict(x=columns_label, y=index_label, color=metric_label),
                        x=pivot_data.columns.astype(str), # Use strings for categorical display if needed, but sorting done numerically
                        y=pivot_data.index.astype(str),
                        color_continuous_scale=colorscale,
                        title=title,
                        aspect="auto")

        # Optional: Add text annotations (can be cluttered)
        # fig.update_traces(text=pivot_data.round(2).astype(str), texttemplate="%{text}")

        fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
        fig.update_xaxes(type='category') # Treat axes as categories after numerical sort
        fig.update_yaxes(type='category')
        _save_figure(fig, filepath)


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH, help=f"Path to CSV (default: {DEFAULT_CSV_PATH})")
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_DIR, help=f"Directory for plots (default: {DEFAULT_PLOT_DIR})")
    parser.add_argument('--fixed-chunk', type=int, default=DEFAULT_FIXED_CHUNK, help=f"Chunk size for vs Threads plots (default: {DEFAULT_FIXED_CHUNK})")
    parser.add_argument('--suffix', type=str, default="", help="Optional suffix for plot filenames")
    parser.add_argument('--all', action='store_true', help="Generate all plot types.")

    # Add flags for specific plot types
    plot_flags = {
        "speedup_vs_threads": "Plot Speedup vs Number of Threads.",
        "time_vs_threads": "Plot Execution Time vs Number of Threads (linear).",
        "time_vs_threads_log": "Plot Execution Time vs Number of Threads (log).",
        "chunk_impact_speedup": "Plot Speedup vs Chunk Size.",
        "chunk_impact_time": "Plot Execution Time vs Chunk Size (linear).",
        # "chunk_impact_time_log": "Plot Execution Time vs Chunk Size (log).", # Merged logic
        "dynamic_chunks_comparison": "Plot Dynamic speedup vs threads for multiple chunk sizes.",
        "blockcyclic_chunks_comparison": "Plot Static Block-Cyclic speedup vs threads for multiple chunk sizes.",
        "heatmaps_speedup": "Plot heatmaps of speedup (Chunk vs Threads).",
        "heatmaps_time": "Plot heatmaps of execution time (Chunk vs Threads)."
        # Consolidated heatmaps are essentially the same as heatmaps, maybe different pivot? Let's keep heatmaps flexible.
    }
    for flag, help_text in plot_flags.items():
        parser.add_argument(f'--{flag.replace("_", "-")}', action='store_true', help=help_text)

    args = parser.parse_args()

    csv_file = Path(args.csv_path)
    plot_dir = Path(args.plot_dir)
    file_suffix = args.suffix

    if not csv_file.is_file():
        print(f"Error: CSV file not found at {csv_file}")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved in: {plot_dir.resolve()}")

    # Load and Preprocess Data centrally
    print(f"Loading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print("Preprocessing data...")
        df['ChunkSize'] = pd.to_numeric(df['ChunkSize'], errors='coerce') # Keep as float initially to handle NaN
        df['ExecutionTimeMs'] = pd.to_numeric(df['ExecutionTimeMs'], errors='coerce')
        df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
        # Handle missing NumThreads (e.g., for Sequential) safely before casting
        df['NumThreads'] = pd.to_numeric(df['NumThreads'], errors='coerce').fillna(1).astype(int)
        # Ensure WorkloadDescription exists and is string
        if 'WorkloadDescription' not in df.columns:
            df['WorkloadDescription'] = "W" + df['WorkloadID'].astype(str) # Fallback description
        else:
            df['WorkloadDescription'] = df['WorkloadDescription'].astype(str)

        # Ensure SchedulerName exists
        if 'SchedulerName' not in df.columns:
             print(f"Error: 'SchedulerName' column not found in {csv_file}")
             return

    except Exception as e:
        print(f"Error loading or preprocessing CSV: {e}")
        return

    # Determine which plots to generate
    generate_all = args.all
    tasks = {
        "speedup_vs_threads": lambda: plot_speedup_vs_threads(df.copy(), plot_dir, args.fixed_chunk, file_suffix),
        "time_vs_threads": lambda: plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, False, file_suffix),
        "time_vs_threads_log": lambda: plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, True, file_suffix + "_log"),
        "chunk_impact_speedup": lambda: plot_chunk_impact(df.copy(), plot_dir, 'Speedup', False, file_suffix),
        "chunk_impact_time": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', False, file_suffix),
        "chunk_impact_time_log": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', True, file_suffix + "_log"),
        "dynamic_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic", file_suffix),
        "blockcyclic_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block-Cyclic", file_suffix),
        "heatmaps_speedup": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', 'ChunkSize', 'NumThreads', file_suffix),
        "heatmaps_time": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'ExecutionTimeMs', 'ChunkSize', 'NumThreads', file_suffix),
         # Example: heatmap pivoted differently (consolidated view)
        "consolidated_heatmaps_speedup": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', 'NumThreads', 'ChunkSize', file_suffix + "_consolidated"),
    }

    any_plot_selected = False
    for flag_name, task in tasks.items():
        arg_name = flag_name.replace("_", "-") # Argument names use hyphens
        if generate_all or getattr(args, flag_name.replace("-", "_"), False): # Check attribute on args using underscores
            task()
            any_plot_selected = True

    if not any_plot_selected:
        print("\nNo plot type selected. Use --all or specific flags like --speedup-vs-threads.")
        parser.print_help()
        return

    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()
