import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import sys
import multiprocessing # Added for CPU count

# --- Patch NumPy bool deprecation ---
if not hasattr(np, 'bool'):
    # np.bool was removed in NumPy 1.24. Use bool instead.
    np.bool = bool # Use Python's built-in bool

# --- Constants & Configuration ---
DEFAULT_CSV_PATH = "../results/performance_results_sencha.csv"
DEFAULT_THEORETICAL_CSV_PATH = "../results/theoretical_speedup.csv" # Added default
DEFAULT_PLOT_DIR = "../results/plots_sencha"
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
HEATMAP_WIDTH = 800
HEATMAP_HEIGHT = 800
MAX_FILTER_CHUNK_SIZE = 1024 # Ignore excessively large chunks in chunk plots

# Define schedulers and colors centrally
SCHEDULER_COLOR_MAP = {
    "Sequential": "black",
    "Static Block": px.colors.qualitative.Plotly[0],
    "Static Cyclic": px.colors.qualitative.Plotly[1],
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic": px.colors.qualitative.Plotly[3]
    # Add other schedulers if they exist in the data
}

# Define which schedulers use the ChunkSize parameter meaningfully
SCHEDULERS_WITH_CHUNK = ["Static Block-Cyclic", "Dynamic"]
# Define schedulers that DO NOT use ChunkSize (or where it's irrelevant/fixed)
SCHEDULERS_NO_CHUNK = ["Sequential", "Static Cyclic", "Static Block"]

# Schedulers specifically for chunk comparison plots (subset of SCHEDULERS_WITH_CHUNK)
CHUNK_SCHEDULER_COLOR_MAP = {
    "Static Block": px.colors.qualitative.Plotly[0],
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic": px.colors.qualitative.Plotly[3]
}

# --- Helper Functions ---

def _prepare_output_dir(base_plot_dir: Path, sub_directory: str) -> Path:
    """Creates the subdirectory within the base plot directory and returns its path."""
    output_dir = base_plot_dir / sub_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _save_figure(fig: go.Figure, filepath: Path):
    """Saves the Plotly figure to a PDF file with error handling."""
    try:
        # Use a consistent PDF export engine
        fig.write_image(filepath, format="pdf", engine="kaleido")
        print(f"  Saved: {filepath.name}") # Print only filename for brevity
    except ValueError as ve:
        if "Full Kaleido" in str(ve):
             print(f"  ERROR saving {filepath.name}: Full Kaleido installation might be required for complex plots. Try 'pip install --upgrade plotly kaleido'.")
        else:
            print(f"  ERROR saving {filepath.name}: {ve}. Ensure Kaleido is installed ('pip install kaleido').")
    except Exception as e:
        print(f"  ERROR saving {filepath.name}: {e}. Ensure Kaleido is installed ('pip install kaleido').")

def _configure_xaxis_ticks(fig: go.Figure, unique_values: List[Any], is_threads: bool):
    """Configures x-axis ticks for threads or chunk sizes."""
    if not unique_values: return

    # Ensure we are working with a Pandas Series before attempting conversion/dropna
    values_series = pd.Series(unique_values)

    # Attempt to convert to numeric for sorting and range calculation
    numeric_series = pd.to_numeric(values_series, errors='coerce')

    # Drop NaN values resulting from conversion errors and get the remaining numeric values
    valid_numeric_values = numeric_series.dropna()

    if valid_numeric_values.empty: # If all values were non-numeric or list was empty
        # Fallback to categorical sorting if no numeric values remain
        try:
             # Sort numerically if possible, treating non-numerics as infinity
             category_array = sorted(unique_values, key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else float('inf'))
        except:
             # Simple string sort as ultimate fallback
             category_array = sorted([str(x) for x in unique_values])
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_array)
        return

    # Convert valid numeric values to a list for further processing
    numeric_values_list = valid_numeric_values.tolist()
    max_val = max(numeric_values_list)
    unique_numeric_sorted = sorted(list(set(numeric_values_list))) # Get unique sorted numerics

    # --- Logic for setting ticks based on numeric values ---
    if len(unique_numeric_sorted) < 8:
        # Use category axis for few distinct numeric values, ensure proper sorting
        # Sort original unique_values based on numeric interpretation
        try:
            category_array = sorted(unique_values, key=lambda x: float(x) if pd.notna(pd.to_numeric(x, errors='coerce')) else float('inf'))
            fig.update_xaxes(type='category', categoryorder='array', categoryarray=[str(c) for c in category_array]) # Use strings for categories
        except: # Fallback if complex sorting fails
             fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted([str(c) for c in unique_values]))

    elif is_threads:
        # Linear scale for threads with specific ticks
        dtick = 2 if max_val <= 16 else 4 if max_val <= 32 else 8
        # Ensure ticks are integers if thread counts are integers
        tickvals = [tick for tick in np.arange(0, max_val + dtick, dtick) if tick >= min(unique_numeric_sorted) or tick == 0]
        ticktext = [str(int(t)) for t in tickvals] # Display as integers
        fig.update_xaxes(type='linear', tickvals=tickvals, ticktext=ticktext, range=[0, max_val * 1.05]) # Extend range slightly

    else: # Chunk sizes or other numeric axes
         # Use linear scale, let Plotly decide ticks unless too dense
         fig.update_xaxes(type='linear')
         # Optional: Improve tick handling for chunk sizes if needed (e.g., force integer ticks)


def _add_amdahl_trace(fig: go.Figure, group_df: pd.DataFrame, threads_col='NumThreads', speedup_col='Speedup'):
    """Calculates and adds Amdahl's Law trace to a speedup plot."""
    if threads_col not in group_df.columns or speedup_col not in group_df.columns:
        print("  Warning: Cannot add Amdahl trace. Missing columns.")
        return

    # Ensure numeric types and filter invalid values
    df_amdahl = group_df[[threads_col, speedup_col]].copy()
    df_amdahl[threads_col] = pd.to_numeric(df_amdahl[threads_col], errors='coerce')
    df_amdahl[speedup_col] = pd.to_numeric(df_amdahl[speedup_col], errors='coerce')
    df_amdahl = df_amdahl.dropna()
    df_amdahl = df_amdahl[df_amdahl[threads_col] >= 1] # Need at least 1 thread

    if df_amdahl.empty:
        print("  Warning: Cannot add Amdahl trace. No valid data points.")
        return

    unique_threads = sorted(df_amdahl[threads_col].unique())
    if not unique_threads or len(unique_threads) < 2: return # Need at least two points for calculation

    max_threads = max(unique_threads)
    if max_threads <= 1: return # No parallelism

    # Find the best speedup achieved at the maximum number of threads *in this group*
    max_thread_data = df_amdahl[df_amdahl[threads_col] == max_threads]
    if max_thread_data.empty: return # Should not happen if checks above passed

    best_speedup_at_max_threads = max_thread_data[speedup_col].max()

    # Also consider the overall best speedup achieved regardless of thread count (if > 1 thread)
    overall_best_speedup = df_amdahl[df_amdahl[threads_col] > 1][speedup_col].max()

    # Use the higher of the two as the basis for 's' calculation, must be > 1
    effective_speedup = max(best_speedup_at_max_threads, overall_best_speedup if pd.notna(overall_best_speedup) else 0)
    effective_threads = max_threads # Use max_threads for calculation

    if pd.isna(effective_speedup) or not np.isfinite(effective_speedup) or effective_speedup <= 1:
        print(f"  Note: Cannot add Amdahl trace. Best speedup ({effective_speedup:.2f}) is not > 1.")
        return

    # Calculate serial fraction 's' based on the effective best speedup
    # S(n) = 1 / (s + (1-s)/n) => s = (n/S(n) - 1) / (n - 1)
    # Avoid division by zero if effective_threads is 1 (already handled)
    s = (effective_threads / effective_speedup - 1) / (effective_threads - 1)
    s = max(0.01, min(0.99, s)) # Clamp s to a reasonable range [0.01, 0.99]

    # Generate points for Amdahl curve
    amdahl_x = np.linspace(1, max_threads, 100)
    amdahl_y = [1 / (s + (1 - s) / n) if n > 0 else 1 for n in amdahl_x]

    fig.add_trace(go.Scatter(
        x=amdahl_x, y=amdahl_y, mode='lines',
        line=dict(color='red', dash='dash', width=1.5),
        name=f"Amdahl's Law (s={s:.2f})", # Using s based on effective best speedup
        showlegend=True
    ))

def _filter_and_sort(df: pd.DataFrame, required_cols: List[str], sort_by: List[str],
                     filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Applies common filtering, NaN dropping, type conversion, and sorting."""
    if df.empty:
        return df

    # Ensure required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        return df_filtered

    # --- Type Conversions and Basic Validity Filters ---
    if 'NumThreads' in df.columns:
        df_filtered['NumThreads'] = pd.to_numeric(df_filtered['NumThreads'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['NumThreads']) # Drop rows where conversion failed
        df_filtered = df_filtered[df_filtered['NumThreads'] >= 1]
        df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    if 'ChunkSize' in df.columns:
        # Ensure ChunkSize is numeric before filtering, handle potential prior NaNs
        df_filtered['ChunkSize'] = pd.to_numeric(df_filtered['ChunkSize'], errors='coerce')
        # Keep NaNs for now, they might be valid (e.g., for Sequential) - filter later if needed
        # Filter out non-positive and excessively large chunk sizes only if they are not NaN
        chunk_filter = (
            (df_filtered['ChunkSize'] > 0) &
            (df_filtered['ChunkSize'] <= MAX_FILTER_CHUNK_SIZE)
        ) | df_filtered['ChunkSize'].isna() # Keep NaNs
        df_filtered = df_filtered[chunk_filter]
        # Attempt to convert to Int64 (nullable integer) if possible, otherwise keep as float
        try:
             df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype('Int64')
        except (TypeError, ValueError):
             pass # Keep as float if conversion fails (e.g., still contains NaNs after filtering?)

    if 'ExecutionTimeMs' in df.columns:
        df_filtered['ExecutionTimeMs'] = pd.to_numeric(df_filtered['ExecutionTimeMs'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['ExecutionTimeMs']) # Drop rows where conversion failed
        df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0] # Time must be positive

    if 'Speedup' in df.columns:
        df_filtered['Speedup'] = pd.to_numeric(df_filtered['Speedup'], errors='coerce')
        # Allow NaN speedup for now, maybe filter later depending on plot

    # --- Apply Custom Filters ---
    if filters:
        for col, value in filters.items():
            if col not in df_filtered.columns:
                print(f"  Warning: Filter column '{col}' not found in DataFrame. Skipping filter.")
                continue
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[col] == value]

    # --- Sorting ---
    valid_sort_cols = [col for col in sort_by if col in df_filtered.columns]
    if valid_sort_cols:
        df_filtered.sort_values(by=valid_sort_cols, inplace=True)

    return df_filtered


# --- Optimal Chunk Size Finding ---

def find_optimal_chunk_sizes(df: pd.DataFrame, default_chunk: int = 64) -> Dict[int, int]:
    """Determines the most performant chunk size for each workload based on mean speedup."""
    print("Finding optimal chunk sizes per workload...")
    optimal_chunks = {}

    # Ensure required columns exist
    if not all(col in df.columns for col in ['WorkloadID', 'SchedulerName', 'ChunkSize', 'Speedup']):
        print("  Warning: Missing required columns for optimal chunk calculation. Returning defaults.")
        # Return default for all unique workload IDs found
        if 'WorkloadID' in df.columns:
             return {wid: default_chunk for wid in df['WorkloadID'].unique()}
        else:
             return {} # Cannot determine workload IDs

    # Filter data relevant for chunk optimization
    df_chunk_opt = df[df['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK)].copy()

    # Use the robust filter function
    try:
        df_filtered = _filter_and_sort(
            df_chunk_opt,
            required_cols=['WorkloadID', 'ChunkSize', 'Speedup', 'NumThreads'], # Need NumThreads > 1
            sort_by=['WorkloadID', 'ChunkSize'],
            filters={'NumThreads': df_chunk_opt[df_chunk_opt['NumThreads'] > 1]['NumThreads'].unique().tolist()} # Only parallel runs matter
        )
        # Specifically drop rows where Speedup or ChunkSize is still NaN after filtering
        df_filtered = df_filtered.dropna(subset=['ChunkSize', 'Speedup'])

    except ValueError as e:
        print(f"  Error during filtering for optimal chunk calculation: {e}. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()}

    if df_filtered.empty:
        print("  Warning: No valid data found for chunk optimization after filtering. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()}

    # Find optimal chunk size for each workload
    for workload_id, group in df_filtered.groupby('WorkloadID'):
        if group.empty:
            optimal_chunks[workload_id] = default_chunk
            continue

        # Group by chunk size and calculate average speedup for this workload
        # Cast ChunkSize to int here if it's Int64 or float, necessary for grouping if mixed types remain
        try:
            group['ChunkSize'] = group['ChunkSize'].astype(int)
            chunk_perf = group.groupby('ChunkSize')['Speedup'].mean().reset_index()
        except Exception as e:
             print(f"  Warning: Could not process chunk sizes for Workload {workload_id}. Using default. Error: {e}")
             optimal_chunks[workload_id] = default_chunk
             continue


        if chunk_perf.empty:
            optimal_chunks[workload_id] = default_chunk
        else:
            # Find the chunk size with the highest average speedup
            best_chunk_row = chunk_perf.loc[chunk_perf['Speedup'].idxmax()]
            best_chunk = int(best_chunk_row['ChunkSize'])
            optimal_chunks[workload_id] = best_chunk
            # print(f"  Workload {workload_id}: optimal chunk size = {best_chunk} (Avg Speedup: {best_chunk_row['Speedup']:.2f})") # Verbose logging

    # Ensure all workloads in the original df have an entry
    all_workload_ids = df['WorkloadID'].unique()
    for wid in all_workload_ids:
        if wid not in optimal_chunks:
            # print(f"  Workload {wid}: No chunk optimization data found. Using default chunk size {default_chunk}.")
            optimal_chunks[wid] = default_chunk

    print(f"Finished finding optimal chunks: {len(optimal_chunks)} workloads processed.")
    return optimal_chunks


# --- Plotting Functions ---

def plot_speedup_vs_threads(df: pd.DataFrame, plot_dir: Path, optimal_chunks: Dict[int, int], file_suffix: str = ""):
    """Generates Speedup vs Number of Threads plots using workload-specific optimal chunk sizes."""
    print("Plotting Speedup vs Threads (using optimal chunks)...")
    output_dir = _prepare_output_dir(plot_dir, "speedup_vs_threads")
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'Speedup', 'ChunkSize']

    # Use a copy for modification
    df_plot_base = df.copy()

    # Ensure optimal_chunks dictionary is not empty
    if not optimal_chunks:
         print("  Warning: Optimal chunks dictionary is empty. Cannot generate optimal speedup plots.")
         return

    all_plots_empty = True
    for workload_id, group_by_workload in df_plot_base.groupby('WorkloadID'):
        optimal_chunk = optimal_chunks.get(workload_id) # Get the optimal chunk for this specific workload
        if optimal_chunk is None:
            print(f"  Warning: No optimal chunk found for Workload {workload_id}. Skipping plot.")
            continue

        # Filter data for this specific workload:
        # 1. Schedulers that DON'T use chunks (ChunkSize is irrelevant or NaN)
        # 2. Schedulers that DO use chunks, selecting ONLY the rows with the optimal chunk size
        df_plot_filtered = group_by_workload[
            group_by_workload['SchedulerName'].isin(SCHEDULERS_NO_CHUNK) |
            (
                group_by_workload['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK) &
                (pd.to_numeric(group_by_workload['ChunkSize'], errors='coerce') == optimal_chunk)
            )
        ].copy()

        # Apply standard filtering and sorting
        try:
            df_plot = _filter_and_sort(df_plot_filtered,
                                       required_cols=['NumThreads', 'Speedup', 'SchedulerName', 'WorkloadDescription'],
                                       sort_by=['SchedulerName', 'NumThreads'])
        except ValueError as e:
             print(f"  Skipping plot for Workload {workload_id} due to filtering error: {e}")
             continue

        if df_plot.empty:
            # print(f"  Skipping plot for Workload {workload_id}: No data after filtering for optimal chunks.")
            continue

        all_plots_empty = False
        workload_desc = df_plot['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Threads - {workload_desc}<br>(Schedulers with best performing chunk size)"
        filename = f"speedup_vs_threads_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(df_plot, x='NumThreads', y='Speedup', color='SchedulerName', markers=True, title=title,
                     labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)'},
                     color_discrete_map=SCHEDULER_COLOR_MAP,
                     category_orders={"SchedulerName": list(SCHEDULER_COLOR_MAP.keys())}) # Ensure consistent legend order

        # Update Sequential trace to use dashed line style in both plot and legend
        fig.update_traces(line=dict(dash='dash'), selector=dict(name='Sequential'))

        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline (Seq)", annotation_position="bottom right")
        unique_threads = sorted(df_plot['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        fig.update_yaxes(rangemode='tozero') # Start y-axis at 0

        # Add Amdahl trace based on the *best performing scheduler* in this filtered data
        if not df_plot.empty:
            best_scheduler_data = df_plot.loc[df_plot.groupby('NumThreads')['Speedup'].idxmax()] # Data points for the best scheduler at each thread count
            _add_amdahl_trace(fig, best_scheduler_data) # Calculate Amdahl based on the envelope curve

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty:
        print("  Warning: No data available to generate any Speedup vs Threads plots with optimal chunks.")


def plot_time_vs_threads(df: pd.DataFrame, plot_dir: Path, optimal_chunks: Dict[int, int], use_log_scale: bool, file_suffix: str = ""):
    """Generates Execution Time vs Number of Threads plots, using optimal chunks."""
    log_suffix = "_log" if use_log_scale else ""
    print(f"Plotting Execution Time vs Threads (log_scale: {use_log_scale}, using optimal chunks)...")
    output_dir = _prepare_output_dir(plot_dir, "time_vs_threads")
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ExecutionTimeMs', 'ChunkSize']

    df_plot_base = df.copy()
    if not optimal_chunks:
         print("  Warning: Optimal chunks dictionary is empty. Cannot generate optimal time plots.")
         return

    all_plots_empty = True
    for workload_id, group_by_workload in df_plot_base.groupby('WorkloadID'):
        optimal_chunk = optimal_chunks.get(workload_id)
        if optimal_chunk is None:
            print(f"  Warning: No optimal chunk found for Workload {workload_id}. Skipping time plot.")
            continue

        # Filter similarly to speedup plot: use optimal chunk where applicable
        df_plot_filtered = group_by_workload[
            group_by_workload['SchedulerName'].isin(SCHEDULERS_NO_CHUNK) |
            (
                group_by_workload['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK) &
                (pd.to_numeric(group_by_workload['ChunkSize'], errors='coerce') == optimal_chunk)
            )
        ].copy()

        # Apply standard filtering and sorting
        try:
            df_plot = _filter_and_sort(df_plot_filtered,
                                      required_cols=['NumThreads', 'ExecutionTimeMs', 'SchedulerName', 'WorkloadDescription'],
                                      sort_by=['SchedulerName', 'NumThreads'])
        except ValueError as e:
             print(f"  Skipping time plot for Workload {workload_id} due to filtering error: {e}")
             continue

        if df_plot.empty:
            # print(f"  Skipping time plot for Workload {workload_id}: No data after filtering.")
            continue

        all_plots_empty = False
        workload_desc = df_plot['WorkloadDescription'].iloc[0]
        y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
        title = f"Execution Time vs Threads - {workload_desc}<br>(Schedulers with best performing chunk size)"
        filename = f"time_vs_threads_W{workload_id}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(df_plot, x='NumThreads', y='ExecutionTimeMs', color='SchedulerName', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=SCHEDULER_COLOR_MAP,
                      category_orders={"SchedulerName": list(SCHEDULER_COLOR_MAP.keys())})

        unique_threads = sorted(df_plot['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        if use_log_scale:
            fig.update_yaxes(rangemode='tozero', tickformat=".1e") # Adjust format for log if needed
        else:
             fig.update_yaxes(rangemode='tozero')

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty:
        print("  Warning: No data available to generate any Time vs Threads plots with optimal chunks.")


def plot_chunk_impact(df: pd.DataFrame, plot_dir: Path, metric: str, use_log_scale: bool = False, file_suffix: str = ""):
    """Generates plots for Speedup or Execution Time vs Chunk Size."""
    metric_col = 'Speedup' if metric.lower() == 'speedup' else 'ExecutionTimeMs'
    metric_label = 'Speedup' if metric.lower() == 'speedup' else 'Execution Time (ms)'
    log_suffix = "_log" if use_log_scale and metric_col == 'ExecutionTimeMs' else ""
    subdir = f"chunk_impact_{metric.lower()}"
    print(f"Plotting {metric_label} vs Chunk Size impact (log_scale: {use_log_scale})...")
    output_dir = _prepare_output_dir(plot_dir, subdir)

    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', metric_col]

    # Filter data first
    try:
        df_plot_base = _filter_and_sort(df.copy(),
                                   required_cols=required_cols,
                                   sort_by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'],
                                   filters={
                                       'SchedulerName': list(CHUNK_SCHEDULER_COLOR_MAP.keys()), # Only schedulers relevant for chunk comparison
                                       'NumThreads': df[df['NumThreads'] > 1]['NumThreads'].unique().tolist() # Only parallel
                                       }
                                  )
        # Drop rows where ChunkSize or the metric is NaN for this specific plot
        df_plot_base = df_plot_base.dropna(subset=['ChunkSize', metric_col])

    except ValueError as e:
        print(f"  Error during filtering for chunk impact plot: {e}. Skipping.")
        return
    except KeyError as e:
         print(f"  Error: Column {e} not found, required for chunk impact plot. Skipping.")
         return


    if df_plot_base.empty:
        print("  Warning: No data available to generate chunk impact plots after filtering.")
        return

    all_plots_empty = True
    # Ensure ChunkSize is integer for grouping/axis labelling if possible
    try:
        df_plot_base['ChunkSize'] = df_plot_base['ChunkSize'].astype(int)
    except (TypeError, ValueError):
        print("  Warning: Could not convert ChunkSize to int for plotting, proceeding with existing type.")


    for (workload_id, num_threads), group in df_plot_base.groupby(['WorkloadID', 'NumThreads']):
        if group.empty: continue

        all_plots_empty = False
        workload_desc = group['WorkloadDescription'].iloc[0]
        y_axis_label = metric_label + (' [Log Scale]' if use_log_scale and metric_col == 'ExecutionTimeMs' else '')
        title = f"{metric_label} vs Chunk Size - {workload_desc}<br>(Threads = {num_threads})"
        filename = f"chunk_{metric.lower()}_W{workload_id}_T{num_threads}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Use category for chunk size axis to handle discrete, potentially non-linear values correctly
        group['ChunkSize_cat'] = group['ChunkSize'].astype(str)
        # Sort categories numerically before plotting
        sorted_chunks_str = sorted(group['ChunkSize'].unique(), key=int)

        fig = px.line(group, x='ChunkSize_cat', y=metric_col, color='SchedulerName', markers=True, title=title,
                      labels={'ChunkSize_cat': 'Chunk Size', metric_col: y_axis_label},
                      log_y=use_log_scale if metric_col == 'ExecutionTimeMs' else False,
                      color_discrete_map=CHUNK_SCHEDULER_COLOR_MAP,
                      category_orders={
                          "SchedulerName": list(CHUNK_SCHEDULER_COLOR_MAP.keys()), # Consistent legend order
                          "ChunkSize_cat": [str(c) for c in sorted_chunks_str] # Ensure x-axis order
                          })

        # Explicitly set category order for x-axis
        fig.update_xaxes(type='category') # categoryorder is handled by category_orders in px.line

        if metric_col == 'Speedup':
            fig.update_yaxes(rangemode='tozero')
        elif use_log_scale:
            fig.update_yaxes(rangemode='tozero', tickformat=".1e")
        else:
            fig.update_yaxes(rangemode='tozero')


        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty:
        print("  Warning: No plots generated for chunk impact (all groups were empty).")


def plot_scheduler_chunk_comparison(df: pd.DataFrame, plot_dir: Path, scheduler_name: str, file_suffix: str = ""):
    """Generates Speedup vs Threads plots for a single scheduler, comparing chunk sizes."""
    print(f"Plotting {scheduler_name} Speedup vs Threads for different chunk sizes...")
    # Sanitize scheduler name for directory
    scheduler_dirname = scheduler_name.lower().replace(' ', '_').replace('-', '_')
    output_dir = _prepare_output_dir(plot_dir, f"{scheduler_dirname}_chunk_comparison")

    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', 'Speedup']

    try:
        df_scheduler = _filter_and_sort(df.copy(),
                                     required_cols=required_cols,
                                     sort_by=['WorkloadID', 'ChunkSize', 'NumThreads'],
                                     filters={
                                         'SchedulerName': scheduler_name,
                                         'NumThreads': df[df['NumThreads'] > 1]['NumThreads'].unique().tolist() # Only parallel
                                         }
                                    )
        # Drop rows where ChunkSize or Speedup is NaN for this specific plot
        df_scheduler = df_scheduler.dropna(subset=['ChunkSize', 'Speedup'])
    except ValueError as e:
        print(f"  Error during filtering for {scheduler_name} chunk comparison: {e}. Skipping.")
        return
    except KeyError as e:
         print(f"  Error: Column {e} not found, required for {scheduler_name} chunk comparison. Skipping.")
         return

    if df_scheduler.empty:
        print(f"  Warning: No data found for scheduler '{scheduler_name}' after filtering. Skipping chunk comparison plot.")
        return

    # Ensure ChunkSize is integer for grouping/axis labelling if possible
    try:
        df_scheduler['ChunkSize'] = df_scheduler['ChunkSize'].astype(int)
    except (TypeError, ValueError):
        print(f"  Warning: Could not convert ChunkSize to int for {scheduler_name} comparison, proceeding with existing type.")

    chunk_sizes = sorted(df_scheduler['ChunkSize'].unique())
    if not chunk_sizes: return

    # Generate distinct colors for chunk sizes using a sequential colorscale
    color_sequence = px.colors.sequential.Viridis
    n_chunks = len(chunk_sizes)
    if n_chunks == 0: return

    # Create a color map ensuring index is within bounds
    color_map = {chunk: color_sequence[min(int(i * (len(color_sequence)-1) / (n_chunks -1 if n_chunks > 1 else 1)), len(color_sequence) - 1)]
                 for i, chunk in enumerate(chunk_sizes)}

    all_plots_empty = True
    for workload_id, group in df_scheduler.groupby('WorkloadID'):
        if group.empty: continue

        all_plots_empty = False
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"{scheduler_name} Speedup vs Threads - {workload_desc}<br>(Comparison of different chunk sizes)"
        filename = f"{scheduler_dirname}_chunks_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Create string representation for coloring/legend
        group['ChunkSize_str'] = "Chunk=" + group['ChunkSize'].astype(str)
        # Ensure order for legend/colors
        chunk_order_str = [f"Chunk={c}" for c in chunk_sizes]

        fig = px.line(group, x='NumThreads', y='Speedup', color='ChunkSize_str', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)', 'ChunkSize_str': 'Chunk Size'},
                      color_discrete_map={f"Chunk={k}": v for k, v in color_map.items()},
                      category_orders={"ChunkSize_str": chunk_order_str} # Control legend order
                     )

        unique_threads = sorted(group['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline (Seq)", annotation_position="bottom right")
        fig.update_yaxes(rangemode='tozero')

        # Add Amdahl trace based on the best speedup achieved across *any chunk size* at max threads
        if not group.empty:
             max_threads_val = group['NumThreads'].max()
             best_overall_data = group.loc[group.groupby('NumThreads')['Speedup'].idxmax()] # Envelope curve across chunks
             _add_amdahl_trace(fig, best_overall_data)

        fig.update_layout(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty:
        print(f"  Warning: No plots generated for {scheduler_name} chunk comparison (all groups were empty).")


def plot_performance_heatmap(df: pd.DataFrame, plot_dir: Path, metric: str, file_suffix: str = ""):
    """Generates heatmaps visualizing scheduler performance (Speedup or Time) vs. Threads and Chunk Size."""
    metric_col = 'Speedup' if metric.lower() == 'speedup' else 'ExecutionTimeMs'
    metric_label = 'Speedup' if metric.lower() == 'speedup' else 'Execution Time (ms)'
    # Higher is better for speedup (Viridis), lower is better for time (Viridis_r)
    colorscale = px.colors.sequential.Viridis if metric_col == 'Speedup' else px.colors.sequential.Viridis_r
    subdir = f"{metric.lower()}_heatmaps_chunk_vs_threads"
    index_label = "Chunk Size"
    columns_label = "Number of Threads"

    print(f"Plotting {metric_label} heatmaps (Chunk Size vs Threads)...")
    output_dir = _prepare_output_dir(plot_dir, subdir)

    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', metric_col]

    try:
        df_heatmap = _filter_and_sort(df.copy(),
                                   required_cols=required_cols,
                                   sort_by=['WorkloadID', 'SchedulerName', 'ChunkSize', 'NumThreads'],
                                   filters={
                                       'SchedulerName': SCHEDULERS_WITH_CHUNK, # Only schedulers where chunk matters
                                       'NumThreads': df[df['NumThreads'] > 1]['NumThreads'].unique().tolist() # Only parallel
                                       }
                                  )
        # Drop rows where essential pivot values are NaN
        df_heatmap = df_heatmap.dropna(subset=['ChunkSize', 'NumThreads', metric_col])
    except ValueError as e:
        print(f"  Error during filtering for heatmap: {e}. Skipping.")
        return
    except KeyError as e:
         print(f"  Error: Column {e} not found, required for heatmap. Skipping.")
         return


    if df_heatmap.empty:
        print("  Warning: No data available to generate heatmaps after filtering.")
        return

    # Ensure ChunkSize and NumThreads are integer for pivoting if possible
    try:
        df_heatmap['ChunkSize'] = df_heatmap['ChunkSize'].astype(int)
        df_heatmap['NumThreads'] = df_heatmap['NumThreads'].astype(int)
    except (TypeError, ValueError):
        print("  Warning: Could not convert ChunkSize/NumThreads to int for heatmap pivoting, proceeding with existing types.")


    all_plots_empty = True
    for (workload_id, scheduler_name), group in df_heatmap.groupby(['WorkloadID', 'SchedulerName']):
        if group.empty or group[['ChunkSize', 'NumThreads']].duplicated().any():
            # Skip if group is empty or if there are duplicate combinations of ChunkSize/NumThreads
            # which would cause pivot_table to aggregate (undesirable if data should be unique)
            if not group.empty:
                 print(f"  Warning: Duplicate ChunkSize/NumThreads combinations found for W{workload_id}, {scheduler_name}. Check data integrity. Skipping heatmap.")
            continue

        workload_desc = group['WorkloadDescription'].iloc[0]
        # Sanitize name for filename
        scheduler_fname = scheduler_name.lower().replace(' ', '_').replace('-', '_')

        try:
            # Use mean aggregation as fallback if duplicates somehow exist, though ideally they shouldn't
            pivot_data = group.pivot_table(index='ChunkSize', columns='NumThreads', values=metric_col, aggfunc='mean')
        except Exception as e:
            print(f"  ERROR creating pivot table for W{workload_id}, {scheduler_name} ({metric}): {e}")
            continue

        if pivot_data.empty: continue

        all_plots_empty = False

        # Sort indices (ChunkSize) and columns (NumThreads) numerically
        pivot_data = pivot_data.sort_index(axis=0, key=lambda x: pd.to_numeric(x))
        pivot_data = pivot_data.sort_index(axis=1, key=lambda x: pd.to_numeric(x))

        title = f"{scheduler_name} {metric_label} Heatmap - {workload_desc}<br>({index_label} vs {columns_label})"
        filename = f"heatmap_{metric.lower()}_{scheduler_fname}_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.imshow(pivot_data,
                        labels=dict(x=columns_label, y=index_label, color=metric_label),
                        # Keep axes types determined by pivot table (should be numeric)
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        color_continuous_scale=colorscale,
                        title=title,
                        aspect="auto", # Adjust aspect ratio automatically
                        text_auto='.2f' # Add values to cells, formatted to 2 decimal places
                        )

        fig.update_traces(hovertemplate=f"{index_label}: %{{y}}<br>{columns_label}: %{{x}}<br>{metric_label}: %{{z:.2f}}<extra></extra>")

        fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
        # Ensure axes are treated appropriately (might be category if pivot index/cols were strings, but we sorted numerically)
        fig.update_xaxes(type='category') # Treat as categories visually after numerical sort
        fig.update_yaxes(type='category')
        _save_figure(fig, filepath)

    if all_plots_empty:
        print("  Warning: No heatmaps generated (all groups were empty or had issues).")


def plot_theoretical_comparison(measured_df, theoretical_df, plot_dir, file_suffix=""):
    """Plots comparing theoretical and actual measured speedup."""
    print("Plotting theoretical vs. measured speedup comparison...")

    if theoretical_df is None or theoretical_df.empty:
        print("  Warning: Theoretical data is missing or empty. Skipping comparison plot.")
        return
    if measured_df is None or measured_df.empty:
         print("  Warning: Measured data is missing or empty. Skipping comparison plot.")
         return

    # Ensure required columns exist
    req_measured = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'Speedup']
    req_theoretical = ['WorkloadID', 'WorkloadDescription', 'Parallelism']
    if not all(col in measured_df.columns for col in req_measured):
        print(f"  Error: Measured data missing columns: {set(req_measured) - set(measured_df.columns)}. Skipping.")
        return
    if not all(col in theoretical_df.columns for col in req_theoretical):
         print(f"  Error: Theoretical data missing columns: {set(req_theoretical) - set(theoretical_df.columns)}. Skipping.")
         return


    # --- Machine Info ---
    try:
        num_cores = multiprocessing.cpu_count()
        machine_info = f"Machine: {num_cores} CPU threads"
    except NotImplementedError:
        machine_info = "Machine: (Could not detect CPU threads)"
    print(f"  {machine_info}")

    output_dir = _prepare_output_dir(plot_dir, "theoretical_comparison")

    # Prepare measured data subset
    measured_subset = measured_df[req_measured].copy()
    # Ensure numeric types before merge/calculation
    measured_subset['NumThreads'] = pd.to_numeric(measured_subset['NumThreads'], errors='coerce')
    measured_subset['Speedup'] = pd.to_numeric(measured_subset['Speedup'], errors='coerce')
    measured_subset = measured_subset.dropna(subset=['WorkloadID', 'NumThreads', 'Speedup'])
    measured_subset['NumThreads'] = measured_subset['NumThreads'].astype(int)


    # Prepare theoretical data subset and ensure numeric types
    theoretical_subset = theoretical_df[req_theoretical].copy()
    theoretical_subset['Parallelism'] = pd.to_numeric(theoretical_subset['Parallelism'], errors='coerce')
    theoretical_subset = theoretical_subset.dropna(subset=['WorkloadID', 'Parallelism'])
    # Keep only one entry per WorkloadID if duplicates exist
    theoretical_subset = theoretical_subset.drop_duplicates(subset=['WorkloadID'], keep='first')


    # Merge data
    # Use outer merge to see if any workloads are missing from either dataset? Or inner to only plot comparable ones. Let's use inner.
    df_joined = pd.merge(
        measured_subset,
        theoretical_subset,
        on=['WorkloadID', 'WorkloadDescription'], # Merge on both if description should match
        how='inner' # Only plot workloads present in both measured and theoretical data
    )

    if df_joined.empty:
        print("  Warning: No matching workloads found between measured and theoretical data after merge. Skipping comparison plots.")
        return

    # Calculate theoretical speedup: min(NumThreads, Parallelism)
    df_joined['Theoretical Speedup'] = df_joined.apply(
        lambda row: min(row['NumThreads'], row['Parallelism']),
        axis=1
    )
    # Rename measured speedup for clarity in legend
    df_joined.rename(columns={'Speedup': 'Measured Speedup'}, inplace=True)

    # Create plots for each workload-scheduler combination
    all_plots_empty = True
    for (workload_id, scheduler_name), group in df_joined.groupby(['WorkloadID', 'SchedulerName']):
        if group.empty: continue

        # Exclude Sequential scheduler from this comparison (its speedup is always 1)
        if scheduler_name == "Sequential":
            continue

        all_plots_empty = False
        workload_desc = group['WorkloadDescription'].iloc[0]
        scheduler_fname = scheduler_name.lower().replace(' ', '_').replace('-', '_')
        title = f"Theoretical vs. Measured Speedup - {workload_desc}<br>Scheduler: {scheduler_name} ({machine_info})"
        filename = f"theory_vs_practice_{scheduler_fname}_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Reshape for plotting (melt)
        plot_data = pd.melt(
            group,
            id_vars=['NumThreads'],
            value_vars=['Measured Speedup', 'Theoretical Speedup'],
            var_name='Speedup Type',
            value_name='Speedup Value'
        )
        plot_data = plot_data.sort_values(by=['Speedup Type', 'NumThreads']) # Ensure lines connect correctly

        fig = px.line(
            plot_data,
            x='NumThreads',
            y='Speedup Value',
            color='Speedup Type',
            markers=True,
            title=title,
            labels={'NumThreads': 'Number of Threads', 'Speedup Value': 'Speedup'},
            color_discrete_map={'Measured Speedup': px.colors.qualitative.Plotly[0], 'Theoretical Speedup': px.colors.qualitative.Plotly[3]}
        )

        # Add ideal linear speedup (y=x) based on unique threads in this group
        x_values = sorted(group['NumThreads'].unique())
        if x_values: # Ensure there are thread values
             # Limit ideal line to max theoretical parallelism if desired? No, usually goes to max threads tested.
             max_thread_in_plot = max(x_values)
             ideal_x = np.linspace(1, max_thread_in_plot, num=max(2,len(x_values))) # Ensure at least 2 points for line
             fig.add_trace(go.Scatter(
                 x=ideal_x,
                 y=ideal_x,
                 mode='lines',
                 line=dict(color='grey', dash='dot'),
                 name='Ideal Linear Speedup'
             ))

        fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5) # Baseline at 1
        _configure_xaxis_ticks(fig, x_values, is_threads=True)
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, legend_title_text='Speedup Type')

        _save_figure(fig, filepath)

    if all_plots_empty:
        print("  Warning: No theoretical comparison plots generated (all groups were empty or skipped).")

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV data.")
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                        help=f"Path to the input CSV file with benchmark results (default: {DEFAULT_CSV_PATH})")
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f"Directory where plot subdirectories will be created (default: {DEFAULT_PLOT_DIR})")
    parser.add_argument('--suffix', type=str, default="",
                        help="Optional suffix to add to all plot filenames (e.g., '_run1')")
    parser.add_argument('--all', action='store_true',
                        help="Generate all standard plot types.")
    # Removed --fixed-chunk as it's not directly used by revised plots relying on optimal_chunks dict
    # parser.add_argument('--fixed-chunk', type=int, help="Override automatic optimal chunk detection with a fixed value.")

    parser.add_argument('--theoretical-csv', type=str, default=DEFAULT_THEORETICAL_CSV_PATH,
                        help=f"Path to CSV with theoretical parallelism data (default: {DEFAULT_THEORETICAL_CSV_PATH})")

    # --- Flags for specific plot types ---
    plot_flags = {
        # Plot name (used in code) : Help text (for argparse)
        "speedup_vs_threads": "Plot Speedup vs Number of Threads (uses optimal chunks).",
        "time_vs_threads": "Plot Execution Time vs Threads (linear scale, uses optimal chunks).",
        "time_vs_threads_log": "Plot Execution Time vs Threads (log scale, uses optimal chunks).",
        "chunk_impact_speedup": "Plot Speedup vs Chunk Size (for chunked schedulers).",
        "chunk_impact_time": "Plot Execution Time vs Chunk Size (linear scale, for chunked schedulers).",
        "chunk_impact_time_log": "Plot Execution Time vs Chunk Size (log scale, for chunked schedulers).",
        "dynamic_chunks_comparison": "Plot Dynamic scheduler speedup vs threads, comparing chunk sizes.",
        "blockcyclic_chunks_comparison": "Plot Static Block-Cyclic speedup vs threads, comparing chunk sizes.",
        "block_chunks_comparison": "Plot Static Block speedup vs threads, comparing chunk sizes.", # Added for block
        "heatmaps_speedup": "Plot heatmaps of speedup (Chunk Size vs Threads).",
        "heatmaps_time": "Plot heatmaps of execution time (Chunk Size vs Threads).",
        "theoretical_comparison": "Plot theoretical vs measured speedup comparison (requires theoretical CSV)."
    }
    # Dynamically add arguments based on plot_flags dictionary
    for flag, help_text in plot_flags.items():
        # Convert flag name from underscore_case to --kebab-case for argparse
        arg_name = f'--{flag.replace("_", "-")}'
        parser.add_argument(arg_name, action='store_true', help=help_text)

    args = parser.parse_args()

    csv_file = Path(args.csv_path).resolve() # Resolve to absolute path
    theoretical_csv_file = Path(args.theoretical_csv).resolve()
    plot_dir = Path(args.plot_dir).resolve()
    file_suffix = args.suffix

    if not csv_file.is_file():
        print(f"Error: Input CSV file not found at {csv_file}")
        sys.exit(1) # Exit if main data file is missing

    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input CSV: {csv_file}")
    print(f"Plot Directory: {plot_dir}")
    if file_suffix:
        print(f"Filename Suffix: '{file_suffix}'")

    # --- Load and Preprocess Measured Data ---
    print("Loading and preprocessing measured data...")
    try:
        df = pd.read_csv(csv_file)

        # --- Basic Column Checks ---
        required_base_cols = ['WorkloadID', 'SchedulerName', 'NumThreads', 'ExecutionTimeMs']
        missing_base = [col for col in required_base_cols if col not in df.columns]
        if missing_base:
             print(f"Error: CSV missing essential columns: {missing_base}")
             sys.exit(1)

        # --- Preprocessing Steps ---
        # Handle potential missing description - create default if needed
        if 'WorkloadDescription' not in df.columns:
            print("  Warning: 'WorkloadDescription' column not found. Creating default from WorkloadID.")
            df['WorkloadDescription'] = "W" + df['WorkloadID'].astype(str)
        else:
            # Ensure it's string and fill potential NaNs
            df['WorkloadDescription'] = df['WorkloadDescription'].astype(str).fillna('Unknown')

        # Ensure 'ChunkSize' exists, even if mostly NaN (important for filters)
        if 'ChunkSize' not in df.columns:
             print("  Warning: 'ChunkSize' column not found. Adding column with NaNs.")
             df['ChunkSize'] = np.nan

        # Ensure 'Speedup' exists, calculate if possible, else fill with NaN
        if 'Speedup' not in df.columns:
            print("  Warning: 'Speedup' column not found. Will attempt calculation if possible, otherwise plots needing Speedup may fail.")
            # Add placeholder - actual calculation might happen later if needed/possible
            df['Speedup'] = np.nan
            # TODO: Consider adding logic here to calculate Speedup based on sequential runs if needed


        # --- Robust Type Conversions ---
        # Use helper function _filter_and_sort for initial cleanup and type conversion
        # Define all columns potentially used by *any* plot
        all_potential_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName',
                              'NumThreads', 'ChunkSize', 'ExecutionTimeMs', 'Speedup']
        df = _filter_and_sort(df, required_cols=required_base_cols, sort_by=['WorkloadID', 'SchedulerName', 'NumThreads', 'ChunkSize'])
        print(f"Data loaded and preprocessed: {len(df)} rows remaining after initial filtering.")

    except FileNotFoundError:
        # This check is redundant due to the is_file() check earlier, but good practice
        print(f"Error: Input CSV file not found at {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input CSV file is empty: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or preprocessing CSV '{csv_file}': {e}")
        sys.exit(1)

    if df.empty:
        print("Error: DataFrame is empty after loading and initial preprocessing. Cannot generate plots.")
        sys.exit(1)

    # --- Determine Optimal Chunk Sizes (Per Workload) ---
    # This is needed for several plot types
    optimal_chunks = find_optimal_chunk_sizes(df.copy()) # Use a copy

    # --- Load Theoretical Data (if needed) ---
    theoretical_df = None
    # Check if theoretical comparison is requested OR if --all is used
    if args.theoretical_comparison or args.all:
        if theoretical_csv_file.is_file():
            print(f"Loading theoretical data from {theoretical_csv_file}...")
            try:
                theoretical_df = pd.read_csv(theoretical_csv_file)
                # Basic validation
                if 'WorkloadID' not in theoretical_df.columns or 'Parallelism' not in theoretical_df.columns:
                     print("  Warning: Theoretical CSV must contain 'WorkloadID' and 'Parallelism' columns.")
                     theoretical_df = None # Invalidate if columns missing
                elif theoretical_df.empty:
                    print("  Warning: Theoretical CSV file is empty.")
                    theoretical_df = None

            except Exception as e:
                print(f"  Error loading theoretical CSV '{theoretical_csv_file}': {e}")
                theoretical_df = None # Ensure it's None on error
        else:
            print(f"Warning: Theoretical data file not found at {theoretical_csv_file}. Skipping theoretical comparison.")
            # Explicitly disable the theoretical plot if file not found, even if --all was used
            args.theoretical_comparison = False


    # --- Define Plotting Tasks ---
    # Use lambda functions to defer execution until needed
    tasks = {
        "speedup_vs_threads": lambda: plot_speedup_vs_threads(df.copy(), plot_dir, optimal_chunks, file_suffix),
        "time_vs_threads": lambda: plot_time_vs_threads(df.copy(), plot_dir, optimal_chunks, False, file_suffix),
        "time_vs_threads_log": lambda: plot_time_vs_threads(df.copy(), plot_dir, optimal_chunks, True, file_suffix),
        # "chunk_impact_speedup": lambda: plot_chunk_impact(df.copy(), plot_dir, 'Speedup', False, file_suffix),
        # "chunk_impact_time": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', False, file_suffix),
        # "chunk_impact_time_log": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', True, file_suffix),
        "dynamic_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic", file_suffix),
        "blockcyclic_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block-Cyclic", file_suffix),
        "block_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block", file_suffix),
        "heatmaps_speedup": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', file_suffix),
        "heatmaps_time": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'ExecutionTimeMs', file_suffix),
        # "theoretical_comparison": lambda: plot_theoretical_comparison(df.copy(), theoretical_df, plot_dir, file_suffix) if theoretical_df is not None else print("Skipping theoretical comparison: No data loaded."),
    }

    # --- Execute Selected Tasks ---
    generate_all = args.all
    any_plot_selected = False
    tasks_executed_count = 0

    print("\n--- Starting Plot Generation ---")
    for task_name, task_func in tasks.items():
        # Check if the corresponding argument flag is set, or if --all is set
        # The attribute name in 'args' uses underscores, matching the task_name
        if generate_all or getattr(args, task_name, False):
            # Special handling for theoretical plot if data failed to load
            if task_name == "theoretical_comparison" and theoretical_df is None and getattr(args, task_name, False):
                 print("Skipping theoretical comparison explicitly requested: Theoretical data file missing or invalid.")
                 continue # Skip this task even if flag was set

            try:
                task_func()
                any_plot_selected = True
                tasks_executed_count += 1
            except Exception as e:
                print(f"\n!!! ERROR generating plots for '{task_name}': {e} !!!")
                # Optionally, print traceback for debugging
                # import traceback
                # traceback.print_exc()
                print("--- Attempting to continue with other plots ---")


    print("\n--- Plot Generation Summary ---")
    if not any_plot_selected and not generate_all:
        print("No specific plot type selected and --all flag not used.")
        parser.print_help()
    elif tasks_executed_count == 0 and (any_plot_selected or generate_all):
         print("No plots were generated. Check warnings and errors above.")
    else:
        print(f"Finished generating {tasks_executed_count} plot type(s).")
        print(f"Plots saved in subdirectories under: {plot_dir}")

    print("------------------------------")


if __name__ == "__main__":
    main()
