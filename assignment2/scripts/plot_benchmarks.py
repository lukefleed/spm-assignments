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
# Deprecated since NumPy 1.20, removed in 1.24. Use standard bool.
# This helps if other parts of the code use np.bool_ directly.
if not hasattr(np, 'bool_'):
    try:
        np.bool_ = bool # Use alias np.bool_ if needed, or just use bool directly
        print("INFO: Manually mapped np.bool_ = bool.")
    except Exception as e:
        print(f"WARNING: Could not map np.bool_ = bool. Error: {e}")

# --- Constants & Configuration ---
DEFAULT_CSV_PATH = "../results/performance_results_sencha.csv"
DEFAULT_THEORETICAL_CSV_PATH = "../results/theoretical_speedup.csv"
DEFAULT_PLOT_DIR = "../results/plots_sencha"
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
HEATMAP_WIDTH = 800
HEATMAP_HEIGHT = 800
MAX_FILTER_CHUNK_SIZE = 1024 # Increased max chunk size based on benchmark args

SCHEDULER_COLOR_MAP = {
    "Sequential": "black",
    "Static Block": px.colors.qualitative.Plotly[0],
    "Static Cyclic": px.colors.qualitative.Plotly[1],
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic TaskQueue": px.colors.qualitative.Plotly[4], # New entry + color
    "Dynamic WorkStealing": px.colors.qualitative.Plotly[3] # Renamed old "Dynamic"
}

# Schedulers that use ChunkSize meaningfully
SCHEDULERS_WITH_CHUNK = [
    "Static Block-Cyclic",
    "Dynamic TaskQueue",
    "Dynamic WorkStealing"
]
# Schedulers that DO NOT use ChunkSize (or where it's N/A)
SCHEDULERS_NO_CHUNK = ["Sequential", "Static Cyclic", "Static Block"]

# Schedulers specifically for the "Chunk Impact" plot (compares perf vs chunk size)
CHUNK_IMPACT_SCHEDULER_COLOR_MAP = {
    "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
    "Dynamic TaskQueue": px.colors.qualitative.Plotly[4],
    "Dynamic WorkStealing": px.colors.qualitative.Plotly[3]
}


def _prepare_output_dir(base_plot_dir: Path, sub_directory: str) -> Path:
    """Creates the subdirectory within the base plot directory and returns its path."""
    output_dir = base_plot_dir / sub_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _save_figure(fig: go.Figure, filepath: Path):
    """Saves the Plotly figure to a PDF file with error handling."""
    try:
        fig.write_image(filepath, format="pdf", engine="kaleido")
        print(f"  Saved: {filepath.name}")
    except ValueError as ve:
        if "Full Kaleido" in str(ve):
             print(f"  ERROR saving {filepath.name}: Full Kaleido installation might be required. Try 'pip install --upgrade plotly kaleido'.")
        else:
            print(f"  ERROR saving {filepath.name}: {ve}. Ensure Kaleido is installed ('pip install kaleido').")
    except Exception as e:
        print(f"  ERROR saving {filepath.name}: {e}. Ensure Kaleido is installed ('pip install kaleido').")

def _configure_xaxis_ticks(fig: go.Figure, unique_values: List[Any], is_threads: bool):
    """Configures x-axis ticks for threads or chunk sizes."""
    if not unique_values: return
    values_series = pd.Series(unique_values)
    numeric_series = pd.to_numeric(values_series, errors='coerce')
    valid_numeric_values = numeric_series.dropna()

    if valid_numeric_values.empty:
        try:
             category_array = sorted(unique_values, key=lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else float('inf'))
        except:
             category_array = sorted([str(x) for x in unique_values])
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_array)
        return

    numeric_values_list = valid_numeric_values.tolist()
    max_val = max(numeric_values_list) if numeric_values_list else 0
    unique_numeric_sorted = sorted(list(set(numeric_values_list)))

    if len(unique_numeric_sorted) < 8:
        try:
            category_array = sorted(unique_values, key=lambda x: float(x) if pd.notna(pd.to_numeric(x, errors='coerce')) else float('inf'))
            fig.update_xaxes(type='category', categoryorder='array', categoryarray=[str(c) for c in category_array])
        except:
             fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted([str(c) for c in unique_values]))
    elif is_threads:
        # Determine dtick based on max_val, ensure it covers the range
        if max_val <= 0: dtick = 1
        elif max_val <= 8: dtick = 1
        elif max_val <= 16: dtick = 2
        elif max_val <= 32: dtick = 4
        elif max_val <= 64: dtick = 8
        else: dtick = 16

        # Generate tick values starting from a reasonable minimum (e.g., 0 or 1 or min value)
        min_tick = 0 # Often start from 0 or 1 for threads
        tickvals = [tick for tick in np.arange(min_tick, max_val + dtick, dtick) if tick >= min(unique_numeric_sorted, default=min_tick) or tick == min_tick]
        # Ensure the max value is included if not covered by dtick steps
        if max_val > 0 and max_val not in tickvals:
            tickvals.append(max_val)
            tickvals.sort()

        ticktext = [str(int(t)) for t in tickvals]
        fig.update_xaxes(type='linear', tickvals=tickvals, ticktext=ticktext, range=[min_tick - max_val * 0.02, max_val * 1.05]) # Slight padding
    else: # Chunk sizes
        # For chunks, linear might be okay, but category might be better if non-uniform steps
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted([str(int(c)) for c in unique_numeric_sorted])) # Use category sorted numerically

def _add_amdahl_trace(fig: go.Figure, group_df: pd.DataFrame, threads_col='NumThreads', speedup_col='Speedup'):
    """Calculates and adds Amdahl's Law trace to a speedup plot."""
    if threads_col not in group_df.columns or speedup_col not in group_df.columns:
        print("  Warning: Cannot add Amdahl trace. Missing columns.")
        return

    df_amdahl = group_df[[threads_col, speedup_col]].copy()
    df_amdahl[threads_col] = pd.to_numeric(df_amdahl[threads_col], errors='coerce')
    df_amdahl[speedup_col] = pd.to_numeric(df_amdahl[speedup_col], errors='coerce')
    df_amdahl = df_amdahl.dropna()
    df_amdahl = df_amdahl[df_amdahl[threads_col] >= 1]

    if df_amdahl.empty:
        print("  Warning: Cannot add Amdahl trace. No valid data points.")
        return

    unique_threads = sorted(df_amdahl[threads_col].unique())
    if not unique_threads or len(unique_threads) < 2: return

    max_threads = max(unique_threads)
    if max_threads <= 1: return

    max_thread_data = df_amdahl[df_amdahl[threads_col] == max_threads]
    if max_thread_data.empty: return

    best_speedup_at_max_threads = max_thread_data[speedup_col].max()
    overall_best_speedup = df_amdahl[df_amdahl[threads_col] > 1][speedup_col].max()

    effective_speedup = max(best_speedup_at_max_threads, overall_best_speedup if pd.notna(overall_best_speedup) else 0)
    effective_threads = max_threads

    if pd.isna(effective_speedup) or not np.isfinite(effective_speedup) or effective_speedup <= 1:
        print(f"  Note: Cannot add Amdahl trace. Best speedup ({effective_speedup:.2f}) is not > 1.")
        return

    # Calculate sequential fraction 's' based on the best observed speedup
    # Avoid division by zero if effective_threads is 1
    s = (effective_threads / effective_speedup - 1) / (effective_threads - 1) if effective_threads > 1 else 0.5
    s = max(0.001, min(0.999, s)) # Clamp s to a reasonable range [0.001, 0.999]

    amdahl_x = np.linspace(1, max_threads, 100)
    amdahl_y = [1 / (s + (1 - s) / n) if n > 0 else 1 for n in amdahl_x]

    fig.add_trace(go.Scatter(
        x=amdahl_x, y=amdahl_y, mode='lines',
        line=dict(color='red', dash='dash', width=1.5),
        name=f"Amdahl (s={s:.3f})",
        showlegend=True
    ))

def _filter_and_sort(df: pd.DataFrame, required_cols: List[str], sort_by: List[str],
                     filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Applies common filtering, NaN dropping, type conversion, and sorting."""
    if df.empty: return df
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")

    df_filtered = df.copy() # Start with a copy

    # Drop NaNs based on required columns, EXCEPT ChunkSize which can be NaN
    cols_to_check_nan = [col for col in required_cols if col != 'ChunkSize']
    df_filtered = df_filtered.dropna(subset=cols_to_check_nan).copy()
    if df_filtered.empty: return df_filtered

    # Convert types and apply specific logic
    if 'NumThreads' in df_filtered.columns:
        df_filtered['NumThreads'] = pd.to_numeric(df_filtered['NumThreads'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['NumThreads'])
        df_filtered = df_filtered[df_filtered['NumThreads'] >= 1]
        df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    if 'ChunkSize' in df_filtered.columns:
        df_filtered['ChunkSize'] = pd.to_numeric(df_filtered['ChunkSize'], errors='coerce')
        # Filter out invalid chunk sizes, but keep intended NaNs for SCHEDULERS_NO_CHUNK
        chunk_filter = (
            (df_filtered['ChunkSize'] > 0) &
            (df_filtered['ChunkSize'] <= MAX_FILTER_CHUNK_SIZE)
        ) | (df_filtered['ChunkSize'].isna() & df_filtered['SchedulerName'].isin(SCHEDULERS_NO_CHUNK))
        df_filtered = df_filtered[chunk_filter]
        try: # Use nullable Int64 for ChunkSize
             df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype('Int64')
        except (TypeError, ValueError): pass # Keep as float if conversion fails (e.g., mixed types remaining)

    if 'ExecutionTimeMs' in df_filtered.columns:
        df_filtered['ExecutionTimeMs'] = pd.to_numeric(df_filtered['ExecutionTimeMs'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['ExecutionTimeMs'])
        df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]

    if 'Speedup' in df_filtered.columns:
        df_filtered['Speedup'] = pd.to_numeric(df_filtered['Speedup'], errors='coerce')
        # Optionally filter out NaN speedups here if needed, depends on plot requirements
        # df_filtered = df_filtered.dropna(subset=['Speedup'])

    # Apply external filters
    if filters:
        for col, value in filters.items():
            if col not in df_filtered.columns:
                print(f"  Warning: Filter column '{col}' not found. Skipping.")
                continue
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[col] == value]

    # Sort the result
    valid_sort_cols = [col for col in sort_by if col in df_filtered.columns]
    if valid_sort_cols and not df_filtered.empty:
        try:
            # Ensure correct types before sorting, especially for ChunkSize
            if 'ChunkSize' in valid_sort_cols and 'Int64' in str(df_filtered['ChunkSize'].dtype):
                 # Sort Int64 correctly (handle potential NaNs)
                 df_filtered = df_filtered.sort_values(by=valid_sort_cols, inplace=False, na_position='last')
            else:
                 df_filtered.sort_values(by=valid_sort_cols, inplace=True, na_position='last')
        except Exception as e:
             print(f"  Warning: Could not sort DataFrame. Error: {e}")

    return df_filtered



def find_optimal_chunk_sizes(df: pd.DataFrame, default_chunk: int = 64) -> Dict[int, int]:
    """Determines the most performant chunk size for each workload based on mean speedup."""
    print("Finding optimal chunk sizes per workload...")
    optimal_chunks = {}
    req_cols_chunk = ['WorkloadID', 'SchedulerName', 'ChunkSize', 'Speedup', 'NumThreads']

    if not all(col in df.columns for col in req_cols_chunk):
        print("  Warning: Missing required columns for optimal chunk calculation. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()} if 'WorkloadID' in df.columns else {}

    df_chunk_opt = df[df['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK)].copy()

    # Filter only parallel runs (threads > 1) and valid numeric chunk/speedup values
    try:
        # Find threads > 1 where chunk optimization is meaningful
        parallel_threads = df_chunk_opt[pd.to_numeric(df_chunk_opt['NumThreads'], errors='coerce') > 1]['NumThreads'].dropna().unique().astype(int).tolist()
        if not parallel_threads:
             print("  Warning: No parallel run data (NumThreads > 1) found for chunk optimization. Returning defaults.")
             return {wid: default_chunk for wid in df['WorkloadID'].unique()}

        # Use _filter_and_sort for consistent processing
        df_filtered = _filter_and_sort(
            df_chunk_opt,
            required_cols=req_cols_chunk,
            sort_by=['WorkloadID', 'ChunkSize'],
            filters={'NumThreads': parallel_threads} # Filter only parallel threads
        )
        # Ensure ChunkSize and Speedup are not NaN for calculation
        df_filtered = df_filtered.dropna(subset=['ChunkSize', 'Speedup'])
        if df_filtered.empty: raise ValueError("No valid data after filtering NaNs in ChunkSize/Speedup.")

        # Convert ChunkSize to int now that NaNs are dropped for grouping
        df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)

    except ValueError as e:
        print(f"  Error during filtering/processing for optimal chunk calculation: {e}. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()}
    except Exception as e: # Catch other potential errors
        print(f"  Unexpected error during chunk optimization setup: {e}. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()}

    if df_filtered.empty:
        print("  Warning: No valid data found for chunk optimization after filtering. Returning defaults.")
        return {wid: default_chunk for wid in df['WorkloadID'].unique()}

    # Group by WorkloadID and calculate best chunk
    for workload_id, group in df_filtered.groupby('WorkloadID'):
        if group.empty:
            optimal_chunks[workload_id] = default_chunk
            continue

        try:
            # Calculate mean speedup for each chunk size within this workload
            # Use all available parallel threads for the average
            chunk_perf = group.groupby('ChunkSize')['Speedup'].mean().reset_index()
        except Exception as e:
             print(f"  Warning: Could not process chunk sizes for Workload {workload_id}. Using default. Error: {e}")
             optimal_chunks[workload_id] = default_chunk
             continue

        if chunk_perf.empty or chunk_perf['Speedup'].isna().all():
            optimal_chunks[workload_id] = default_chunk
        else:
            # Find the chunk size corresponding to the maximum mean speedup
            best_chunk_row = chunk_perf.loc[chunk_perf['Speedup'].idxmax()]
            best_chunk = int(best_chunk_row['ChunkSize'])
            optimal_chunks[workload_id] = best_chunk
            # print(f"  Workload {workload_id}: Optimal chunk {best_chunk} (Avg Speedup: {best_chunk_row['Speedup']:.2f})")

    # Ensure all original workloads have an entry, using default if not calculated
    all_workload_ids = df['WorkloadID'].unique()
    for wid in all_workload_ids:
        if wid not in optimal_chunks:
            optimal_chunks[wid] = default_chunk
            # print(f"  Workload {wid}: Using default chunk {default_chunk} (no valid data for optimization).")

    print(f"Finished finding optimal chunks: {len(optimal_chunks)} workloads processed.")
    return optimal_chunks



def plot_speedup_vs_threads(df: pd.DataFrame, plot_dir: Path, optimal_chunks: Dict[int, int], file_suffix: str = ""):
    """Generates Speedup vs Number of Threads plots using workload-specific optimal chunk sizes."""
    print("Plotting Speedup vs Threads (using optimal chunks)...")
    output_dir = _prepare_output_dir(plot_dir, "speedup_vs_threads")
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'Speedup', 'ChunkSize']
    df_plot_base = df.copy()
    if not optimal_chunks: print("  Warning: Optimal chunks dict empty. Cannot proceed."); return

    all_plots_empty = True
    processed_workloads = 0
    for workload_id, group_by_workload in df_plot_base.groupby('WorkloadID'):
        optimal_chunk = optimal_chunks.get(workload_id)
        if optimal_chunk is None:
            # This case should ideally be handled by find_optimal_chunk_sizes returning a default
            print(f"  Warning: No optimal chunk found or defaulted for WorkloadID {workload_id}. Skipping plot.")
            continue

        # Filter for:
        # 1. Schedulers that DO NOT use ChunkSize (ChunkSize is likely NaN or irrelevant)
        # 2. Schedulers that DO use ChunkSize AND their ChunkSize matches the optimal one for this workload
        df_plot_filtered = group_by_workload[
            group_by_workload['SchedulerName'].isin(SCHEDULERS_NO_CHUNK) |
            (
                group_by_workload['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK) &
                # Coerce ChunkSize to numeric, compare with optimal_chunk (int)
                (pd.to_numeric(group_by_workload['ChunkSize'], errors='coerce') == optimal_chunk)
            )
        ].copy()

        try:
            df_plot = _filter_and_sort(df_plot_filtered,
                                       required_cols=['NumThreads', 'Speedup', 'SchedulerName', 'WorkloadDescription'],
                                       sort_by=['SchedulerName', 'NumThreads'])
        except ValueError as e:
            print(f"  Warning: Filtering error for Workload {workload_id}: {e}. Skipping plot.")
            continue # Skip plot if filtering error

        if df_plot.empty:
            # print(f"  Note: No data points remained for Workload {workload_id} after filtering for optimal chunk {optimal_chunk}. Skipping plot.")
            continue

        # Check if we have both Sequential and at least one parallel scheduler result
        schedulers_in_plot = df_plot['SchedulerName'].unique()
        if "Sequential" not in schedulers_in_plot or len(schedulers_in_plot) < 2:
            # print(f"  Note: Not enough data diversity for Workload {workload_id} (needs Seq + parallel). Skipping plot.")
            continue


        all_plots_empty = False
        processed_workloads += 1
        workload_desc = df_plot['WorkloadDescription'].iloc[0] if not df_plot.empty else f"Workload {workload_id}"
        title = f"Speedup vs Threads - {workload_desc}<br>(Chunked Schedulers using Optimal Chunk Size: {optimal_chunk})"
        filename = f"speedup_vs_threads_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(df_plot, x='NumThreads', y='Speedup', color='SchedulerName', markers=True, title=title,
                     labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)'},
                     color_discrete_map=SCHEDULER_COLOR_MAP,
                     category_orders={"SchedulerName": list(SCHEDULER_COLOR_MAP.keys())}) # Ensure consistent order

        fig.update_traces(line=dict(dash='dash'), selector=dict(name='Sequential')) # Dash Sequential line
        fig.add_hline(y=1.0, line_dash="dot", line_color="grey", annotation_text="Baseline (Seq=1x)", annotation_position="bottom right")
        unique_threads = sorted(df_plot['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        fig.update_yaxes(rangemode='tozero') # Ensure y-axis starts at 0

        # Add Amdahl trace based on the best performing scheduler at each thread count in this specific plot
        if not df_plot[df_plot['NumThreads'] > 1].empty:
             best_scheduler_data = df_plot.loc[df_plot.groupby('NumThreads')['Speedup'].idxmax()]
             _add_amdahl_trace(fig, best_scheduler_data)

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty: print("  Warning: No data available for Speedup vs Threads plots after filtering.")
    elif processed_workloads > 0: print(f"  Generated {processed_workloads} Speedup vs Threads plots.")


def plot_time_vs_threads(df: pd.DataFrame, plot_dir: Path, optimal_chunks: Dict[int, int], use_log_scale: bool, file_suffix: str = ""):
    """Generates Execution Time vs Number of Threads plots, using optimal chunks."""
    log_suffix = "_log" if use_log_scale else ""
    print(f"Plotting Execution Time vs Threads (log: {use_log_scale}, using optimal chunks)...")
    output_dir = _prepare_output_dir(plot_dir, "time_vs_threads")
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ExecutionTimeMs', 'ChunkSize']
    df_plot_base = df.copy()
    if not optimal_chunks: print("  Warning: Optimal chunks dict empty. Cannot proceed."); return

    all_plots_empty = True
    processed_workloads = 0
    for workload_id, group_by_workload in df_plot_base.groupby('WorkloadID'):
        optimal_chunk = optimal_chunks.get(workload_id)
        if optimal_chunk is None:
            print(f"  Warning: No optimal chunk found or defaulted for WorkloadID {workload_id}. Skipping plot.")
            continue

        # Filter similar to speedup plot
        df_plot_filtered = group_by_workload[
            group_by_workload['SchedulerName'].isin(SCHEDULERS_NO_CHUNK) |
            (
                group_by_workload['SchedulerName'].isin(SCHEDULERS_WITH_CHUNK) &
                (pd.to_numeric(group_by_workload['ChunkSize'], errors='coerce') == optimal_chunk)
            )
        ].copy()

        try:
            df_plot = _filter_and_sort(df_plot_filtered,
                                      required_cols=['NumThreads', 'ExecutionTimeMs', 'SchedulerName', 'WorkloadDescription'],
                                      sort_by=['SchedulerName', 'NumThreads'])
        except ValueError as e:
             print(f"  Warning: Filtering error for Workload {workload_id}: {e}. Skipping plot.")
             continue

        if df_plot.empty:
            # print(f"  Note: No data points remained for Workload {workload_id} after filtering for optimal chunk {optimal_chunk}. Skipping plot.")
            continue

        # Check if we have results for multiple schedulers
        schedulers_in_plot = df_plot['SchedulerName'].unique()
        if len(schedulers_in_plot) < 1: # Need at least one scheduler
            # print(f"  Note: Not enough data for Workload {workload_id}. Skipping plot.")
            continue

        all_plots_empty = False
        processed_workloads += 1
        workload_desc = df_plot['WorkloadDescription'].iloc[0] if not df_plot.empty else f"Workload {workload_id}"
        y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
        title = f"Execution Time vs Threads - {workload_desc}<br>(Chunked Schedulers using Optimal Chunk Size: {optimal_chunk})"
        filename = f"time_vs_threads_W{workload_id}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(df_plot, x='NumThreads', y='ExecutionTimeMs', color='SchedulerName', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=SCHEDULER_COLOR_MAP,
                      category_orders={"SchedulerName": list(SCHEDULER_COLOR_MAP.keys())})

        unique_threads = sorted(df_plot['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        if use_log_scale: fig.update_yaxes(rangemode='tozero', tickformat=".1e") # Scientific notation for log scale
        else: fig.update_yaxes(rangemode='tozero')

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty: print("  Warning: No data available for Time vs Threads plots after filtering.")
    elif processed_workloads > 0: print(f"  Generated {processed_workloads} Time vs Threads plots.")


def plot_chunk_impact(df: pd.DataFrame, plot_dir: Path, metric: str, use_log_scale: bool = False, file_suffix: str = ""):
    """Generates plots for Speedup or Execution Time vs Chunk Size for relevant schedulers."""
    metric_col = 'Speedup' if metric.lower() == 'speedup' else 'ExecutionTimeMs'
    metric_label = 'Speedup' if metric.lower() == 'speedup' else 'Execution Time (ms)'
    log_suffix = "_log" if use_log_scale and metric_col == 'ExecutionTimeMs' else ""
    subdir = f"chunk_impact_{metric.lower()}"
    print(f"Plotting {metric_label} vs Chunk Size impact (log_scale: {use_log_scale})...")
    output_dir = _prepare_output_dir(plot_dir, subdir)
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', metric_col]

    relevant_schedulers = list(CHUNK_IMPACT_SCHEDULER_COLOR_MAP.keys())

    try:
        df_filtered_base = df[
            (pd.to_numeric(df['NumThreads'], errors='coerce') > 1) &
            (df['SchedulerName'].isin(relevant_schedulers))
        ].copy()

        if df_filtered_base.empty:
            print("  Warning: No parallel run data found for relevant chunked schedulers. Skipping chunk impact plots.")
            return

        # Get the list of parallel threads present in the filtered data
        parallel_threads = sorted(df_filtered_base['NumThreads'].dropna().unique().astype(int).tolist())
        if not parallel_threads:
             print("  Warning: No valid parallel thread counts found after initial filtering. Skipping chunk impact plots.")
             return

        df_plot_base = _filter_and_sort(
           df_filtered_base,
           required_cols=required_cols,
           sort_by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'],
           filters={ # Apply filters again within _filter_and_sort
               'SchedulerName': relevant_schedulers,
               'NumThreads': parallel_threads
           }
        )
        df_plot_base = df_plot_base.dropna(subset=['ChunkSize', metric_col])
        df_plot_base['ChunkSize'] = df_plot_base['ChunkSize'].astype(int)

    except ValueError as e: print(f"  Error filtering for chunk impact: {e}. Skipping."); return
    except KeyError as e: print(f"  Error: Column {e} missing for chunk impact. Skipping."); return
    except Exception as e: print(f"  Unexpected error during chunk impact setup: {e}. Skipping."); return

    if df_plot_base.empty: print("  Warning: No data available for chunk impact plots after filtering."); return

    all_plots_empty = True
    processed_groups = 0
    for (workload_id, num_threads), group in df_plot_base.groupby(['WorkloadID', 'NumThreads']):
        if group.empty: continue
        # Ensure there's variation in chunk sizes for the plot to be meaningful
        if group['ChunkSize'].nunique() < 2: continue

        all_plots_empty = False
        processed_groups += 1
        workload_desc = group['WorkloadDescription'].iloc[0] if not group.empty else f"Workload {workload_id}"
        y_axis_label = metric_label + (' [Log Scale]' if use_log_scale and metric_col == 'ExecutionTimeMs' else '')
        title = f"{metric_label} vs Chunk Size - {workload_desc}<br>(Threads = {num_threads})"
        filename = f"chunk_{metric.lower()}_W{workload_id}_T{num_threads}{log_suffix}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Ensure ChunkSize is treated categorically and sorted numerically for the x-axis
        group['ChunkSize_cat'] = group['ChunkSize'].astype(str)
        sorted_chunks_str = sorted(group['ChunkSize'].unique(), key=int) # Sort numerically before converting to str

        fig = px.line(group, x='ChunkSize_cat', y=metric_col, color='SchedulerName', markers=True, title=title,
                      labels={'ChunkSize_cat': 'Chunk Size', metric_col: y_axis_label},
                      log_y=use_log_scale if metric_col == 'ExecutionTimeMs' else False,
                      color_discrete_map=CHUNK_IMPACT_SCHEDULER_COLOR_MAP,
                      category_orders={
                          "SchedulerName": list(CHUNK_IMPACT_SCHEDULER_COLOR_MAP.keys()), # Use defined order
                          "ChunkSize_cat": [str(c) for c in sorted_chunks_str] # Use numerically sorted order
                          })

        # Explicitly set x-axis type to category AFTER sorting
        fig.update_xaxes(type='category')
        # Configure y-axis range
        if metric_col == 'Speedup': fig.update_yaxes(rangemode='tozero')
        elif use_log_scale: fig.update_yaxes(rangemode='tozero', tickformat=".1e")
        else: fig.update_yaxes(rangemode='tozero')

        fig.update_layout(legend_title_text='Scheduler', width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty: print(f"  Warning: No plots generated for {metric_label} vs Chunk Size impact (data might lack chunk size variation).")
    elif processed_groups > 0: print(f"  Generated {processed_groups} {metric_label} vs Chunk Size plots.")


def plot_scheduler_chunk_comparison(df: pd.DataFrame, plot_dir: Path, scheduler_name: str, file_suffix: str = ""):
    """Generates Speedup vs Threads plots for a single scheduler, comparing chunk sizes."""
    print(f"Plotting {scheduler_name} Speedup vs Threads for different chunk sizes...")
    scheduler_dirname = scheduler_name.lower().replace(' ', '_').replace('-', '_')
    output_dir = _prepare_output_dir(plot_dir, f"{scheduler_dirname}_chunk_comparison")
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', 'Speedup']

    try:
        # Filter data for the specific scheduler and parallel runs first
        df_filtered_base = df[
            (df['SchedulerName'] == scheduler_name) &
            (pd.to_numeric(df['NumThreads'], errors='coerce') > 1) # Compare chunks only for parallel runs
        ].copy()

        if df_filtered_base.empty:
            print(f"  Warning: No parallel run data found for '{scheduler_name}'. Skipping comparison plot.")
            return

        # Use _filter_and_sort for consistency
        df_scheduler = _filter_and_sort(
            df_filtered_base,
            required_cols=required_cols,
            sort_by=['WorkloadID', 'ChunkSize', 'NumThreads'],
            filters={'SchedulerName': scheduler_name} # Redundant but safe
        )
        # Ensure required columns for the plot are not NaN
        df_scheduler = df_scheduler.dropna(subset=['ChunkSize', 'Speedup', 'NumThreads'])
        if df_scheduler.empty: raise ValueError("No valid data after filtering NaNs.")
        # Convert ChunkSize and NumThreads to int after dropping NaNs
        df_scheduler['ChunkSize'] = df_scheduler['ChunkSize'].astype(int)
        df_scheduler['NumThreads'] = df_scheduler['NumThreads'].astype(int)

    except ValueError as e: print(f"  Error filtering/processing data for {scheduler_name}: {e}. Skipping plot."); return
    except KeyError as e: print(f"  Error: Column {e} missing for {scheduler_name}. Skipping plot."); return
    except Exception as e: print(f"  Unexpected error during {scheduler_name} setup: {e}. Skipping plot."); return

    if df_scheduler.empty: print(f"  Warning: No data available for '{scheduler_name}' after filtering. Skipping plot."); return

    chunk_sizes = sorted(df_scheduler['ChunkSize'].unique())
    if not chunk_sizes or len(chunk_sizes) < 2:
        print(f"  Warning: Not enough chunk size variation ({len(chunk_sizes)}) for '{scheduler_name}'. Skipping comparison plot.")
        return

    # Create a color map for the chunk sizes
    color_sequence = px.colors.sequential.Viridis # Or another sequential scale like Plasma, Inferno
    n_chunks = len(chunk_sizes)
    color_map = {chunk: color_sequence[min(int(i * (len(color_sequence)-1) / (n_chunks -1 if n_chunks > 1 else 1)), len(color_sequence) - 1)]
                 for i, chunk in enumerate(chunk_sizes)}

    all_plots_empty = True
    processed_workloads = 0
    for workload_id, group in df_scheduler.groupby('WorkloadID'):
        if group.empty: continue
        # Check if there's data for more than one chunk size in this workload
        if group['ChunkSize'].nunique() < 2: continue

        all_plots_empty = False
        processed_workloads += 1
        workload_desc = group['WorkloadDescription'].iloc[0] if not group.empty else f"Workload {workload_id}"
        title = f"{scheduler_name} Speedup vs Threads - {workload_desc}<br>(Comparison of Chunk Sizes)"
        filename = f"{scheduler_dirname}_chunks_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Create a string column for coloring/legend, ensuring correct sorting
        group['ChunkSize_str'] = "Chunk=" + group['ChunkSize'].astype(str)
        chunk_order_str = [f"Chunk={c}" for c in chunk_sizes] # Use the globally sorted chunk_sizes

        fig = px.line(group, x='NumThreads', y='Speedup', color='ChunkSize_str', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup', 'ChunkSize_str': 'Chunk Size'},
                      color_discrete_map={f"Chunk={k}": v for k, v in color_map.items()},
                      category_orders={"ChunkSize_str": chunk_order_str} # Order legend by chunk size
                     )

        unique_threads = sorted(group['NumThreads'].unique())
        _configure_xaxis_ticks(fig, unique_threads, is_threads=True)
        fig.add_hline(y=1.0, line_dash="dot", line_color="grey", annotation_text="Baseline (Seq=1x)", annotation_position="bottom right")
        fig.update_yaxes(rangemode='tozero')

        # Add Amdahl trace based on the best performance across all chunks *within this plot*
        if not group.empty:
             best_overall_data = group.loc[group.groupby('NumThreads')['Speedup'].idxmax()]
             _add_amdahl_trace(fig, best_overall_data)

        fig.update_layout(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        _save_figure(fig, filepath)

    if all_plots_empty: print(f"  Warning: No plots generated for {scheduler_name} chunk comparison (data might lack chunk size variation per workload).")
    elif processed_workloads > 0: print(f"  Generated {processed_workloads} {scheduler_name} chunk comparison plots.")


def plot_performance_heatmap(df: pd.DataFrame, plot_dir: Path, metric: str, scheduler_name: str, file_suffix: str = ""):
    """Generates heatmaps of Speedup or Execution Time (Chunk Size vs Threads) for a specific scheduler."""
    metric_col = 'Speedup' if metric.lower() == 'speedup' else 'ExecutionTimeMs'
    metric_label = 'Speedup' if metric.lower() == 'speedup' else 'Execution Time (ms)'
    # Use Viridis for Speedup (higher is better), Viridis_r (reversed) for Time (lower is better)
    colorscale = px.colors.sequential.Viridis if metric_col == 'Speedup' else px.colors.sequential.Viridis_r
    scheduler_fname = scheduler_name.lower().replace(' ', '_').replace('-', '_')
    subdir = f"{metric.lower()}_heatmaps_chunk_vs_threads"
    index_label, columns_label = "Chunk Size", "Number of Threads"

    print(f"Plotting {metric_label} heatmaps for {scheduler_name} (Chunk Size vs Threads)...")
    output_dir = _prepare_output_dir(plot_dir, subdir)
    required_cols = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'ChunkSize', metric_col]

    try:
        # Filter for the specific scheduler and parallel runs first
        df_filtered_base = df[
            (df['SchedulerName'] == scheduler_name) &
            (pd.to_numeric(df['NumThreads'], errors='coerce') > 1) # Heatmap makes sense for parallel runs
        ].copy()

        if df_filtered_base.empty:
            print(f"  Warning: No parallel run data found for '{scheduler_name}'. Skipping heatmaps.")
            return

        # Use _filter_and_sort for consistency
        df_heatmap = _filter_and_sort(
            df_filtered_base,
            required_cols=required_cols,
            sort_by=['WorkloadID', 'ChunkSize', 'NumThreads'],
            filters={'SchedulerName': scheduler_name}
        )
        # Drop rows where essential heatmap axes or values are NaN
        df_heatmap = df_heatmap.dropna(subset=['ChunkSize', 'NumThreads', metric_col])
        if df_heatmap.empty: raise ValueError("No valid data after filtering NaNs.")
        # Convert axes to int *after* dropping NaNs
        df_heatmap['ChunkSize'] = df_heatmap['ChunkSize'].astype(int)
        df_heatmap['NumThreads'] = df_heatmap['NumThreads'].astype(int)

    except ValueError as e: print(f"  Error filtering/processing data for {scheduler_name} heatmap: {e}. Skipping."); return
    except KeyError as e: print(f"  Error: Column {e} missing for {scheduler_name} heatmap. Skipping."); return
    except Exception as e: print(f"  Unexpected error during {scheduler_name} heatmap setup: {e}. Skipping."); return

    if df_heatmap.empty: print(f"  Warning: No data available for {scheduler_name} heatmaps after filtering."); return

    all_plots_empty = True
    processed_workloads = 0
    for workload_id, group in df_heatmap.groupby(['WorkloadID']): # Group only by workload now
        if group.empty: continue

        # Ensure variation in both axes
        if group['ChunkSize'].nunique() < 2 or group['NumThreads'].nunique() < 2:
            continue

        workload_desc = group['WorkloadDescription'].iloc[0] if not group.empty else f"Workload {workload_id}"
        try:
            # Create pivot table, averaging results if multiple runs exist for the same config
            pivot_data = group.pivot_table(index='ChunkSize', columns='NumThreads', values=metric_col, aggfunc='mean')
        except Exception as e: print(f"  ERROR creating pivot table for W{workload_id}, {scheduler_name}: {e}"); continue

        if pivot_data.empty: continue

        all_plots_empty = False
        processed_workloads += 1
        # Sort index (ChunkSize) and columns (NumThreads) numerically
        pivot_data = pivot_data.sort_index(axis=0) # Sort rows (Chunk Size)
        pivot_data = pivot_data.sort_index(axis=1) # Sort columns (Num Threads)

        title = f"{scheduler_name} {metric_label} Heatmap - {workload_desc}<br>({index_label} vs {columns_label})"
        filename = f"heatmap_{metric.lower()}_{scheduler_fname}_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.imshow(pivot_data,
                        labels=dict(x=columns_label, y=index_label, color=metric_label),
                        x=pivot_data.columns, # Use sorted columns
                        y=pivot_data.index,   # Use sorted index
                        color_continuous_scale=colorscale,
                        title=title,
                        aspect="auto")       # Use equal aspect ratio to ensure cells are square

        fig.update_traces(hovertemplate=f"{index_label}: %{{y}}<br>{columns_label}: %{{x}}<br>{metric_label}: %{{z:.2f}}<extra></extra>")
        fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)

        # Show alternating labels on x-axis to prevent overcrowding
        columns = pivot_data.columns.tolist()
        tickvals = pivot_data.columns

        # Only show every second label when we have more than 8 thread values
        if len(columns) > 8:
            ticktext = []
            for i, t in enumerate(columns):
                # Show label for every even index (0, 2, 4...) to avoid overcrowding
                if i % 2 == 0:
                    ticktext.append(str(int(t)))
                else:
                    ticktext.append(" ")  # Empty string for odd indices
        else:
            # If not many threads, show all labels
            ticktext = [str(int(t)) for t in columns]

        # Update x-axis with alternating labels
        fig.update_xaxes(type='linear', tickvals=tickvals, ticktext=ticktext)

        # Keep y-axis as is, showing all chunk size labels
        fig.update_yaxes(type='linear', tickvals=pivot_data.index, ticktext=[str(int(c)) for c in pivot_data.index])

        _save_figure(fig, filepath)

    if all_plots_empty: print(f"  Warning: No {scheduler_name} heatmaps generated (data might lack variation or failed pivot).")
    elif processed_workloads > 0: print(f"  Generated {processed_workloads} {scheduler_name} {metric_label} heatmaps.")

def plot_theoretical_comparison(measured_df, theoretical_df, plot_dir, file_suffix=""):
    """Plots comparing theoretical maximum speedup (based on parallelism) and actual measured speedup."""
    print("Plotting theoretical vs. measured speedup comparison...")
    if theoretical_df is None or theoretical_df.empty: print("  Warning: Theoretical data missing or empty. Skipping comparison."); return
    if measured_df is None or measured_df.empty: print("  Warning: Measured data missing or empty. Skipping comparison."); return

    # --- Validate Input DataFrames ---
    req_measured = ['WorkloadID', 'WorkloadDescription', 'SchedulerName', 'NumThreads', 'Speedup']
    req_theoretical = ['WorkloadID', 'WorkloadDescription', 'Parallelism'] # Expecting these columns

    missing_measured = [c for c in req_measured if c not in measured_df.columns]
    missing_theoretical = [c for c in req_theoretical if c not in theoretical_df.columns]

    if missing_measured: print(f"Error: Measured data missing required columns: {missing_measured}. Skipping comparison."); return
    if missing_theoretical: print(f"Error: Theoretical data missing required columns: {missing_theoretical}. Skipping comparison."); return

    try: num_cores = multiprocessing.cpu_count(); machine_info = f"Machine: {num_cores} Cores"
    except NotImplementedError: machine_info = "Machine: (Unknown Cores)"
    print(f"  {machine_info}")
    output_dir = _prepare_output_dir(plot_dir, "theoretical_comparison")

    # --- Prepare Measured Data ---
    try:
        measured_subset = measured_df[req_measured].copy()
        measured_subset['NumThreads'] = pd.to_numeric(measured_subset['NumThreads'], errors='coerce')
        measured_subset['Speedup'] = pd.to_numeric(measured_subset['Speedup'], errors='coerce')
        # Drop rows where key numeric values are invalid
        measured_subset = measured_subset.dropna(subset=['WorkloadID', 'NumThreads', 'Speedup'])
        measured_subset = measured_subset[measured_subset['NumThreads'] >= 1] # Ensure valid thread counts
        measured_subset['NumThreads'] = measured_subset['NumThreads'].astype(int)
        # Exclude sequential runs from the "measured speedup" lines in the comparison plot,
        # but keep the data for potential joins if needed later.
        measured_parallel = measured_subset[measured_subset['SchedulerName'] != "Sequential"].copy()
        if measured_parallel.empty: print("  Warning: No non-Sequential measured data found. Cannot generate comparison plots."); return
    except Exception as e: print(f"  Error preparing measured data: {e}. Skipping comparison."); return

    # --- Prepare Theoretical Data ---
    try:
        theoretical_subset = theoretical_df[req_theoretical].copy()
        theoretical_subset['Parallelism'] = pd.to_numeric(theoretical_subset['Parallelism'], errors='coerce')
        # Drop rows where key values are invalid
        theoretical_subset = theoretical_subset.dropna(subset=['WorkloadID', 'Parallelism'])
        theoretical_subset = theoretical_subset[theoretical_subset['Parallelism'] >= 1] # Parallelism must be >= 1
        # Ensure unique WorkloadID-Parallelism pairs, keeping the first description if duplicates exist
        theoretical_subset = theoretical_subset.drop_duplicates(subset=['WorkloadID'], keep='first')
    except Exception as e: print(f"  Error preparing theoretical data: {e}. Skipping comparison."); return

    if theoretical_subset.empty: print("  Warning: No valid theoretical data found after cleaning. Cannot generate comparison plots."); return

    # --- Join Data ---
    # Use the parallel measured data for joining
    df_joined = pd.merge(measured_parallel, theoretical_subset, on=['WorkloadID', 'WorkloadDescription'], how='inner')
    if df_joined.empty: print("  Warning: No matching workloads found between measured (parallel) and theoretical data after cleaning. Skipping plots."); return

    # --- Calculate Theoretical Speedup Limit ---
    # Theoretical speedup = min(NumThreads, Parallelism) -> Cannot exceed available threads or inherent parallelism
    df_joined['Theoretical Speedup'] = df_joined.apply(lambda row: min(row['NumThreads'], row['Parallelism']), axis=1)
    df_joined.rename(columns={'Speedup': 'Measured Speedup'}, inplace=True) # Rename for clarity in plot

    # --- Plotting ---
    all_plots_empty = True
    processed_groups = 0
    # Group by Workload and Scheduler to create one plot per combination
    for (workload_id, scheduler_name), group in df_joined.groupby(['WorkloadID', 'SchedulerName']):
        if group.empty: continue
        # Need multiple thread counts to make a meaningful line plot
        if group['NumThreads'].nunique() < 2: continue

        all_plots_empty = False
        processed_groups += 1
        workload_desc = group['WorkloadDescription'].iloc[0] if not group.empty else f"Workload {workload_id}"
        scheduler_fname = scheduler_name.lower().replace(' ', '_').replace('-', '_')
        title = f"Theoretical vs. Measured Speedup - {workload_desc}<br>Scheduler: {scheduler_name} ({machine_info})"
        filename = f"theory_vs_practice_{scheduler_fname}_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Prepare data for Plotly: Melt to have 'Speedup Type' column
        plot_data = pd.melt(group,
                            id_vars=['NumThreads'],
                            value_vars=['Measured Speedup', 'Theoretical Speedup'],
                            var_name='Speedup Type', value_name='Speedup Value')
        # Sort for consistent line drawing
        plot_data = plot_data.sort_values(by=['Speedup Type', 'NumThreads'])

        fig = px.line(plot_data, x='NumThreads', y='Speedup Value', color='Speedup Type', markers=True, title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup Value': 'Speedup'},
                      color_discrete_map={'Measured Speedup': SCHEDULER_COLOR_MAP.get(scheduler_name, px.colors.qualitative.Plotly[0]), # Use scheduler color
                                          'Theoretical Speedup': 'grey'}) # Theoretical line color

        # Add Ideal Linear Speedup line (y=x)
        x_values = sorted(group['NumThreads'].unique())
        if x_values:
             max_thread_in_plot = max(x_values)
             # Extend ideal line slightly beyond max threads if needed
             ideal_x = np.linspace(1, max(max_thread_in_plot, group['Parallelism'].iloc[0]), num=100)
             ideal_y = ideal_x
             fig.add_trace(go.Scatter(x=ideal_x, y=ideal_y, mode='lines', line=dict(color='black', dash='dot', width=1), name='Ideal (Linear) Speedup'))

        # Add baseline y=1
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5)

        _configure_xaxis_ticks(fig, x_values, is_threads=True)
        # Set y-axis limit based on max of theoretical parallelism, measured speedup, and ideal speedup
        max_y = max(group['Parallelism'].max(), group['Measured Speedup'].max(), max_thread_in_plot if x_values else 1)
        fig.update_yaxes(rangemode='tozero', range=[0, max_y * 1.1]) # Start at 0, add some padding

        fig.update_layout(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, legend_title_text='Speedup Type')
        _save_figure(fig, filepath)

    if all_plots_empty: print("  Warning: No theoretical comparison plots generated (data might lack thread variation or matching workloads).")
    elif processed_groups > 0: print(f"  Generated {processed_groups} theoretical comparison plots.")


def main():
    # Explicitly map np.bool/np.bool_ to Python's bool before pandas.read_csv
    # This is needed because pandas' internal C parsers might expect these types,
    # which were deprecated/removed in NumPy 1.24+.
    import numpy as np
    if not hasattr(np, 'bool'):
        try:
            np.bool = bool # Map the removed np.bool
            print("INFO: Manually mapped np.bool = bool for pandas compatibility.")
        except Exception as e:
            print(f"WARNING: Could not map np.bool = bool. Pandas parsing might fail. Error: {e}")
    # Also ensure np.bool_ exists (this is handled by the global patch too, but belt-and-suspenders)
    if not hasattr(np, 'bool_'):
        try:
            np.bool_ = bool # Map the deprecated np.bool_
            print("INFO: Manually mapped np.bool_ = bool inside main.")
        except Exception as e:
            print(f"WARNING: Could not map np.bool_ = bool inside main. Error: {e}")
    # --- END NumPy/Pandas Compatibility PATCH ---

    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV data.")
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                        help=f"Path to the input CSV file with benchmark results (default: {DEFAULT_CSV_PATH})")
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f"Directory where plot subdirectories will be created (default: {DEFAULT_PLOT_DIR})")
    parser.add_argument('--suffix', type=str, default="",
                        help="Optional suffix to add to all plot filenames (e.g., '_run1')")
    parser.add_argument('--all', action='store_true',
                        help="Generate all standard plot types.")
    parser.add_argument('--theoretical-csv', type=str, default=DEFAULT_THEORETICAL_CSV_PATH,
                        help=f"Path to CSV with theoretical parallelism data (default: {DEFAULT_THEORETICAL_CSV_PATH})")

    # --- Flags for specific plot types ---
    plot_flags = {
        # Basic Comparisons (using optimal chunks)
        "speedup_vs_threads": "Plot Speedup vs Threads (uses optimal chunks).",
        "time_vs_threads": "Plot Execution Time vs Threads (linear scale, uses optimal chunks).",
        "time_vs_threads_log": "Plot Execution Time vs Threads (log scale, uses optimal chunks).",
        # Chunk Impact Comparison (Multiple Schedulers)
        "chunk_impact_speedup": "Plot Speedup vs Chunk Size.",
        "chunk_impact_time": "Plot Exec Time vs Chunk Size (linear scale).",
        "chunk_impact_time_log": "Plot Exec Time vs Chunk Size (log scale).",
        # Specific Scheduler Chunk Comparisons
        "blockcyclic_chunks_comparison": "Plot Static Block-Cyclic speedup vs threads, comparing chunk sizes.",
        "dynamic_taskqueue_chunks_comparison": "Plot Dynamic TaskQueue speedup vs threads, comparing chunk sizes.",
        "dynamic_workstealing_chunks_comparison": "Plot Dynamic WorkStealing speedup vs threads, comparing chunk sizes.",
        # Heatmaps (Per Scheduler)
        "heatmaps_speedup_blockcyclic": "Plot heatmaps of speedup for Static Block-Cyclic.",
        # "heatmaps_time_blockcyclic": "Plot heatmaps of execution time for Static Block-Cyclic.",
        "heatmaps_speedup_taskqueue": "Plot heatmaps of speedup for Dynamic TaskQueue.",
        # "heatmaps_time_taskqueue": "Plot heatmaps of execution time for Dynamic TaskQueue.",
        "heatmaps_speedup_workstealing": "Plot heatmaps of speedup for Dynamic WorkStealing.",
        # "heatmaps_time_workstealing": "Plot heatmaps of execution time for Dynamic WorkStealing.",
        # Theoretical
        "theoretical_comparison": "Plot theoretical vs measured speedup comparison."
    }
    any_plot_selected = False # Track if user selected any specific plot
    for flag, help_text in plot_flags.items():
        arg_name = f'--{flag.replace("_", "-")}'
        parser.add_argument(arg_name, action='store_true', help=help_text)

    args = parser.parse_args()

    # Check if any specific plot flag was set
    for flag in plot_flags:
        if getattr(args, flag.replace("-", "_"), False):
            any_plot_selected = True
            break

    csv_file = Path(args.csv_path).resolve()
    theoretical_csv_file = Path(args.theoretical_csv).resolve()
    plot_dir = Path(args.plot_dir).resolve()
    file_suffix = args.suffix

    if not csv_file.is_file(): print(f"Error: Input CSV not found: {csv_file}"); sys.exit(1)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input CSV: {csv_file}")
    print(f"Plot Directory: {plot_dir}")
    if file_suffix: print(f"Filename Suffix: '{file_suffix}'")

    # --- Load and Preprocess Measured Data ---
    print("Loading and preprocessing measured data...")
    try:
        # The patch above should handle the np.bool/bool8 issue for read_csv
        df = pd.read_csv(csv_file)

        # --- Post-load Validation and Type Conversion ---
        required_base_cols = ['WorkloadID', 'SchedulerName', 'NumThreads', 'ExecutionTimeMs']
        missing_base = [col for col in required_base_cols if col not in df.columns]
        if missing_base:
            print(f"Error: CSV missing essential columns: {missing_base}")
            sys.exit(1)

        # Add missing optional columns with default values if necessary
        if 'WorkloadDescription' not in df.columns:
            print("  Warning: 'WorkloadDescription' missing, creating default."); df['WorkloadDescription'] = "W" + df['WorkloadID'].astype(str)
        else: # Ensure it's string and handle potential NaNs
            df['WorkloadDescription'] = df['WorkloadDescription'].astype(str).fillna('Unknown')

        if 'ChunkSize' not in df.columns:
            print("  Warning: 'ChunkSize' missing, adding column with NaN."); df['ChunkSize'] = np.nan
        else: # Convert ChunkSize, coercing errors to NaN
             df['ChunkSize'] = pd.to_numeric(df['ChunkSize'], errors='coerce')

        if 'Speedup' not in df.columns:
            print("  Warning: 'Speedup' missing, adding column with NaN."); df['Speedup'] = np.nan
        else: # Convert Speedup, coercing errors to NaN
            df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')

        # Convert core numeric columns, coercing errors
        df['NumThreads'] = pd.to_numeric(df['NumThreads'], errors='coerce')
        df['ExecutionTimeMs'] = pd.to_numeric(df['ExecutionTimeMs'], errors='coerce')

        # Initial filtering and sorting (will be refined in plot functions)
        # Use a broader set of potentially required columns here
        initial_req_cols = list(set(required_base_cols + ['WorkloadDescription', 'ChunkSize', 'Speedup']))
        df = _filter_and_sort(df,
                              required_cols=initial_req_cols,
                              sort_by=['WorkloadID', 'SchedulerName', 'NumThreads', 'ChunkSize'])

        print(f"Data loaded: {len(df)} rows after initial filter and type conversion.")
        if df.empty: print("Error: DataFrame empty after initial loading and filtering."); sys.exit(1)

    except Exception as e:
        print(f"Error loading/preprocessing CSV '{csv_file}': {e}")
        # Provide more context for the common bool8 error
        if "bool8" in str(e) or "bool" in str(e):
            print("\n >> This might be related to NumPy/Pandas version incompatibility.")
            print(" >> Ensure NumPy < 1.24 or that the compatibility patch is working.")
        sys.exit(1)


    # --- Determine Optimal Chunk Sizes ---
    # Pass a copy to avoid modifying the main df
    optimal_chunks = find_optimal_chunk_sizes(df.copy())

    # --- Load Theoretical Data ---
    theoretical_df = None
    # Load if needed for the specific flag or --all
    if args.theoretical_comparison or args.all:
        if theoretical_csv_file.is_file():
            print(f"Loading theoretical data from {theoretical_csv_file}...")
            try:
                theoretical_df = pd.read_csv(theoretical_csv_file)
                # Basic validation
                if 'WorkloadID' not in theoretical_df.columns or 'Parallelism' not in theoretical_df.columns:
                    print("  Warning: Theoretical CSV must contain 'WorkloadID' and 'Parallelism' columns. Disabling theoretical plots."); theoretical_df = None
                elif theoretical_df.empty:
                    print("  Warning: Theoretical CSV is empty. Disabling theoretical plots."); theoretical_df = None
                else:
                    # Ensure WorkloadDescription exists for joining consistency
                    if 'WorkloadDescription' not in theoretical_df.columns:
                         print("  Warning: Theoretical CSV missing 'WorkloadDescription', attempting merge on WorkloadID only.")
                         # If merging only on ID, ensure types match measured_df
                         theoretical_df['WorkloadID'] = pd.to_numeric(theoretical_df['WorkloadID'], errors='coerce')
                    else:
                        theoretical_df['WorkloadDescription'] = theoretical_df['WorkloadDescription'].astype(str).fillna('Unknown')

            except Exception as e: print(f"  Error loading theoretical CSV: {e}. Disabling theoretical plots."); theoretical_df = None
        else:
            print(f"Warning: Theoretical file not found: {theoretical_csv_file}. Disabling theoretical plots.")
            # Ensure flag is false if file not found, even if user requested it
            args.theoretical_comparison = False


    # --- Define Plotting Tasks ---
    # Use lambda functions to delay execution until needed
    tasks = {
        # Basic Comparisons (using optimal chunks)
        "speedup_vs_threads": lambda: plot_speedup_vs_threads(df.copy(), plot_dir, optimal_chunks, file_suffix),
        "time_vs_threads": lambda: plot_time_vs_threads(df.copy(), plot_dir, optimal_chunks, False, file_suffix),
        "time_vs_threads_log": lambda: plot_time_vs_threads(df.copy(), plot_dir, optimal_chunks, True, file_suffix),
        # Chunk Impact Comparison (Multiple Schedulers)
        # "chunk_impact_speedup": lambda: plot_chunk_impact(df.copy(), plot_dir, 'Speedup', False, file_suffix),
        # "chunk_impact_time": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', False, file_suffix),
        # "chunk_impact_time_log": lambda: plot_chunk_impact(df.copy(), plot_dir, 'ExecutionTimeMs', True, file_suffix),
        # Specific Scheduler Chunk Comparisons
        "blockcyclic_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block-Cyclic", file_suffix),
        # "block_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block", file_suffix), # Often less interesting as chunk is N/A
        "dynamic_taskqueue_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic TaskQueue", file_suffix),
        "dynamic_workstealing_chunks_comparison": lambda: plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic WorkStealing", file_suffix),
        # Heatmaps (Per Scheduler) - Only for schedulers where chunk matters
        "heatmaps_speedup_blockcyclic": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', "Static Block-Cyclic", file_suffix),
        "heatmaps_time_blockcyclic": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'ExecutionTimeMs', "Static Block-Cyclic", file_suffix),
        "heatmaps_speedup_taskqueue": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', "Dynamic TaskQueue", file_suffix),
        "heatmaps_time_taskqueue": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'ExecutionTimeMs', "Dynamic TaskQueue", file_suffix),
        "heatmaps_speedup_workstealing": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'Speedup', "Dynamic WorkStealing", file_suffix),
        "heatmaps_time_workstealing": lambda: plot_performance_heatmap(df.copy(), plot_dir, 'ExecutionTimeMs', "Dynamic WorkStealing", file_suffix),
        # Theoretical
        # "theoretical_comparison": lambda: plot_theoretical_comparison(df.copy(), theoretical_df, plot_dir, file_suffix) if theoretical_df is not None else print("Skipping theoretical plot: No valid theoretical data loaded."),
    }

    # --- Execute Selected Tasks ---
    generate_all = args.all
    tasks_executed_count = 0
    print("\n--- Starting Plot Generation ---")
    for task_name, task_func in tasks.items():
        run_task = generate_all or getattr(args, task_name, False)

        if run_task:
            # Special check for theoretical comparison needing data
            if task_name == "theoretical_comparison" and theoretical_df is None:
                 print("Skipping theoretical comparison: Data missing or invalid.")
                 continue # Skip this task specifically

            try:
                task_func() # Execute the plotting function
                tasks_executed_count += 1
            except Exception as e:
                # Catch errors during specific plot generation and report, but try to continue
                import traceback
                print(f"\n!!! ERROR generating plots for '{task_name}': {e} !!!")
                print(traceback.format_exc()) # Print full traceback for debugging
                print("--- Attempting to continue with next plot type ---")

    print("\n--- Plot Generation Summary ---")
    # Determine if any plotting action was intended
    intended_action = generate_all or any_plot_selected
    if not intended_action:
        print("No specific plot type selected and --all flag not used. Use --help for options.")
        parser.print_help()
    elif tasks_executed_count == 0 and intended_action:
         print("No plots were successfully generated. Please check warnings and errors above.")
    else:
        print(f"Finished generating {tasks_executed_count} plot type(s).")
        print(f"Plots saved in subdirectories under: {plot_dir}")
    print("------------------------------")

if __name__ == "__main__":
    main()
