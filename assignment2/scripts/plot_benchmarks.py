import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # For adding horizontal lines (baseline)
import os
import argparse
import numpy as np # For NaN representation
from pathlib import Path

# --- Constants ---
DEFAULT_CSV_PATH = "../results/performance_results.csv"
DEFAULT_PLOT_DIR = "../results/plots"
# Chunk size used for Speedup/Time vs Threads plots
# Choose a representative value or one that yielded good results
DEFAULT_FIXED_CHUNK = 64

# --- Plotting Functions ---

def plot_speedup_vs_threads(df, plot_dir, fixed_chunk_size, file_suffix="", width=1000, height=600):
    """Generates Speedup vs Number of Threads plots for each workload.

    Compares schedulers at a fixed chunk size (for those that use it).
    Includes Sequential and Static Cyclic for reference. Adds Amdahl's Law curve.
    """
    print(f"Plotting Speedup vs Threads (fixed chunk: {fixed_chunk_size})...")
    output_dir = plot_dir / "speedup_vs_threads"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for fixed chunk size OR schedulers that don't use chunk size
    schedulers_no_chunk = ["Sequential", "Static Cyclic"]

    # Ensure clean data for relevant columns
    df_filtered = df.dropna(subset=['NumThreads', 'Speedup'])
    df_filtered = df_filtered[df_filtered['NumThreads'] >= 1] # Ensure valid thread counts
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_plot = df_filtered[
        (df_filtered['ChunkSize'] == fixed_chunk_size) |
        (df_filtered['SchedulerName'].isin(schedulers_no_chunk))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Sort for correct line plotting
    df_plot.sort_values(by=['WorkloadID', 'SchedulerName', 'NumThreads'], inplace=True)

    # Consistent color mapping
    color_map = {
        "Sequential": "black",
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Cyclic": px.colors.qualitative.Plotly[1],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for workload_id, group in df_plot.groupby('WorkloadID'):
        if group.empty: continue # Skip empty groups
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"speedup_vs_threads_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group,
                      x='NumThreads',
                      y='Speedup',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)'},
                      color_discrete_map=color_map)

        # Add baseline speedup line
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")

        # Configure X-axis ticks based on the number of unique thread counts
        unique_threads = sorted(group['NumThreads'].unique())
        if not unique_threads: continue
        if len(unique_threads) < 8:
             fig.update_xaxes(type='category', categoryorder='array', categoryarray=unique_threads)
        else:
             fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4) # Adjust tick frequency

        # Estimate sequential fraction (s) using Amdahl's Law inversion
        max_threads = max(unique_threads) if unique_threads else 0
        if max_threads > 1:
            max_thread_data = group[group['NumThreads'] == max_threads]
            if not max_thread_data.empty:
                best_speedup = max_thread_data['Speedup'].max()
                # Check if best_speedup is valid (not NaN or infinite) and > 0
                if pd.notna(best_speedup) and np.isfinite(best_speedup) and best_speedup > 0 and best_speedup != 1:
                    # S(n) = 1 / (s + (1-s)/n) => s = (n/S(n) - 1) / (n - 1)
                    # Handle edge case n=1 separately, s is undefined. Check n != 1 already done.
                    s = (max_threads / best_speedup - 1) / (max_threads - 1)
                    # Clamp s to a reasonable range [0.01, 0.99] to avoid extreme/invalid values
                    s = max(0.01, min(0.99, s))

                    # Generate Amdahl's curve points
                    amdahl_x = np.linspace(1, max_threads, 100) # Smooth curve
                    amdahl_y = [1 / (s + (1-s)/n) if n > 0 else 1 for n in amdahl_x] # Handle n=0 case if necessary

                    fig.add_trace(go.Scatter(
                        x=amdahl_x, y=amdahl_y, mode='lines',
                        line=dict(color='red', dash='dash', width=1.5),
                        name=f"Amdahl's Law (s={s:.2f})", showlegend=True
                    ))

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}. Ensure Kaleido is installed ('pip install kaleido').")


def plot_time_vs_threads(df, plot_dir, fixed_chunk_size, use_log_scale=True, file_suffix="", width=800, height=600):
    """Generates Execution Time vs Number of Threads plots for each workload."""
    print(f"Plotting Execution Time vs Threads (fixed chunk: {fixed_chunk_size}, log_scale: {use_log_scale})...")
    output_dir = plot_dir / "time_vs_threads"
    output_dir.mkdir(parents=True, exist_ok=True)

    schedulers_no_chunk = ["Sequential", "Static Cyclic"]
    df_filtered = df.dropna(subset=['NumThreads', 'ExecutionTimeMs'])
    df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0] # Ignore errors or non-positive times
    df_filtered = df_filtered[df_filtered['NumThreads'] >= 1]
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)


    df_plot = df_filtered[
        (df_filtered['ChunkSize'] == fixed_chunk_size) |
        (df_filtered['SchedulerName'].isin(schedulers_no_chunk))
    ].copy()

    df_plot.sort_values(by=['WorkloadID', 'SchedulerName', 'NumThreads'], inplace=True)

    color_map = {
        "Sequential": "black",
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Cyclic": px.colors.qualitative.Plotly[1],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for workload_id, group in df_plot.groupby('WorkloadID'):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
        title = f"Exec Time vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"time_vs_threads_W{workload_id}{'_log' if use_log_scale else ''}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group,
                      x='NumThreads',
                      y='ExecutionTimeMs',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'NumThreads': 'Number of Threads', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=color_map)

        unique_threads = sorted(group['NumThreads'].unique())
        if not unique_threads: continue
        if len(unique_threads) < 8:
             fig.update_xaxes(type='category', categoryorder='array', categoryarray=unique_threads)
        else:
             fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4)

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")

def plot_chunk_impact_speedup(df, plot_dir, file_suffix="", width=800, height=600):
    """Generates Speedup vs Chunk Size plots for each workload and thread count."""
    print("Plotting Speedup vs Chunk Size impact...")
    output_dir = plot_dir / "chunk_impact_speedup"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Consider only schedulers using chunk size > 0 and valid results
    # Static Block might ignore chunk size in some implementations; included for now.
    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', 'Speedup'])
    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0] # Only numeric chunk sizes > 0
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]
    df_filtered = df_filtered[df_filtered['NumThreads'] > 1] # Only parallel runs
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_filtered.sort_values(by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'], inplace=True)

    # Specific colors for relevant schedulers
    color_map = {
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for (workload_id, num_threads), group in df_filtered.groupby(['WorkloadID', 'NumThreads']):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Chunk Size - Workload: {workload_desc}<br>(Threads = {num_threads})"
        filename = f"chunk_speedup_W{workload_id}_T{num_threads}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Use category for chunk size axis as they are discrete values
        group['ChunkSize_cat'] = group['ChunkSize'].astype(str) # Use string for category axis

        fig = px.line(group,
                      x='ChunkSize_cat',
                      y='Speedup',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'ChunkSize_cat': 'Chunk Size', 'Speedup': 'Speedup'},
                      color_discrete_map=color_map)

        # Ensure chunks are ordered correctly on the category axis
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted(group['ChunkSize'].unique()))

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")


def plot_chunk_impact_time(df, plot_dir, use_log_scale=True, file_suffix="", width=800, height=600):
    """Generates Execution Time vs Chunk Size plots for each workload and thread count."""
    print(f"Plotting Execution Time vs Chunk Size impact (log_scale: {use_log_scale})...")
    output_dir = plot_dir / "chunk_impact_time"
    output_dir.mkdir(parents=True, exist_ok=True)

    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', 'ExecutionTimeMs'])
    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0]
    df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]
    df_filtered = df_filtered[df_filtered['NumThreads'] > 1] # Only parallel runs
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_filtered.sort_values(by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'], inplace=True)

    color_map = {
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for (workload_id, num_threads), group in df_filtered.groupby(['WorkloadID', 'NumThreads']):
         if group.empty: continue
         workload_desc = group['WorkloadDescription'].iloc[0]
         y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
         title = f"Exec Time vs Chunk Size - Workload: {workload_desc}<br>(Threads = {num_threads})"
         filename = f"chunk_time_W{workload_id}_T{num_threads}{'_log' if use_log_scale else ''}{file_suffix}.pdf"
         filepath = output_dir / filename

         group['ChunkSize_cat'] = group['ChunkSize'].astype(str)

         fig = px.line(group,
                      x='ChunkSize_cat',
                      y='ExecutionTimeMs',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'ChunkSize_cat': 'Chunk Size', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=color_map)

         fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted(group['ChunkSize'].unique()))
         fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
         try:
             fig.write_image(filepath, format="pdf")
             print(f"  Saved: {filepath}")
         except Exception as e:
             print(f"  ERROR saving {filepath}: {e}")


def plot_scheduler_chunk_comparison(df, plot_dir, scheduler_name, chunk_sizes=[16, 32, 64, 96, 128, 256], file_suffix="", width=1000, height=600):
    """Generates Speedup vs Threads plots for a single scheduler with multiple lines per chunk size."""
    print(f"Plotting {scheduler_name} Speedup vs Threads for different chunk sizes...")
    output_dir = plot_dir / f"{scheduler_name.lower().replace(' ', '_')}_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter dataframe for the specified scheduler and chunk sizes
    df_scheduler = df[(df['SchedulerName'] == scheduler_name) & (df['ChunkSize'].isin(chunk_sizes))].copy()

    # Convert types and sort
    df_scheduler['NumThreads'] = df_scheduler['NumThreads'].astype(int)
    df_scheduler = df_scheduler[df_scheduler['NumThreads'] > 1] # Only parallel
    df_scheduler['ChunkSize'] = df_scheduler['ChunkSize'].astype(int)
    df_scheduler.sort_values(by=['WorkloadID', 'ChunkSize', 'NumThreads'], inplace=True)

    # Generate distinct colors for chunk sizes using a sequential colorscale
    color_sequence = px.colors.sequential.Viridis
    n_chunks = len(chunk_sizes)
    if n_chunks == 0: return # Skip if no matching chunks
    # Map each chunk size to a color in the sequence
    color_map = {chunk: color_sequence[min(i * len(color_sequence) // n_chunks, len(color_sequence)-1)]
                 for i, chunk in enumerate(sorted(chunk_sizes))}

    for workload_id, group in df_scheduler.groupby('WorkloadID'):
        if group.empty: continue
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"{scheduler_name} Speedup vs Threads - Workload: {workload_desc}<br>(Comparison of different chunk sizes)"
        filename = f"{scheduler_name.lower().replace(' ', '_')}_chunks_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Convert ChunkSize to string for a clearer legend
        group['ChunkSize_str'] = "Chunk=" + group['ChunkSize'].astype(str)

        fig = px.line(group,
                      x='NumThreads',
                      y='Speedup',
                      color='ChunkSize_str', # Use string for readable legend
                      markers=True,
                      title=title,
                      labels={'NumThreads': 'Number of Threads',
                              'Speedup': 'Speedup (relative to Sequential)',
                              'ChunkSize_str': 'Chunk Size'},
                      color_discrete_map={f"Chunk={k}": v for k, v in color_map.items()} # Apply color map
                      )

        # Configure X-axis ticks
        unique_threads = sorted(group['NumThreads'].unique())
        if not unique_threads: continue
        if len(unique_threads) < 8:
            fig.update_xaxes(type='category', categoryorder='array', categoryarray=unique_threads)
        else:
            fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4)

        # Add baseline speedup line
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")

        # Estimate Amdahl's Law curve based on best performance across chunks
        max_threads = max(unique_threads) if unique_threads else 0
        if max_threads > 1:
            max_thread_data = group[group['NumThreads'] == max_threads]
            if not max_thread_data.empty:
                best_speedup = max_thread_data['Speedup'].max()
                if pd.notna(best_speedup) and np.isfinite(best_speedup) and best_speedup > 0 and best_speedup != 1:
                    s = (max_threads / best_speedup - 1) / (max_threads - 1)
                    s = max(0.01, min(0.99, s)) # Clamp

                    amdahl_x = np.linspace(1, max_threads, 100)
                    amdahl_y = [1 / (s + (1-s)/n) if n > 0 else 1 for n in amdahl_x]
                    fig.add_trace(go.Scatter(
                        x=amdahl_x, y=amdahl_y, mode='lines',
                        line=dict(color='red', dash='dash', width=1.5),
                        name=f"Amdahl's Law (s={s:.2f})", showlegend=True
                    ))

        fig.update_layout(width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")


def plot_scheduler_heatmaps(df, plot_dir, show_speedup=True, file_suffix="", width=800, height=800):
    """Generates heatmaps visualizing scheduler performance vs. thread count and chunk size."""
    print(f"Plotting scheduler performance heatmaps ({'speedup' if show_speedup else 'execution time'})...")
    output_dir = plot_dir / "scheduler_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Consider only schedulers using chunk size
    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]

    # Prepare data
    metric = 'Speedup' if show_speedup else 'ExecutionTimeMs'
    metric_label = 'Speedup' if show_speedup else 'Execution Time (ms)'

    # Filter valid data
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', metric])
    if not show_speedup: # Filter non-positive times if plotting time
        df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]

    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0] # Only numeric chunks > 0
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]
    df_filtered = df_filtered[df_filtered['NumThreads'] > 1] # Only parallel runs

    # Convert types
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    # Generate heatmap for each workload and relevant scheduler
    for workload_id, workload_group in df_filtered.groupby('WorkloadID'):
        if workload_group.empty: continue
        workload_desc = workload_group['WorkloadDescription'].iloc[0]

        for scheduler_name, scheduler_group in workload_group.groupby('SchedulerName'):
            if scheduler_group.empty: continue
            # Pivot data to create a matrix: ChunkSize (rows) x NumThreads (columns)
            try:
                pivot_data = scheduler_group.pivot_table(
                    index='ChunkSize', columns='NumThreads', values=metric, aggfunc='mean' # Use mean for potential duplicates
                )
            except Exception as e:
                print(f"  ERROR creating pivot table for W{workload_id}, {scheduler_name}: {e}")
                continue

            if pivot_data.empty: continue

            # Sort indices for better display
            pivot_data = pivot_data.sort_index(axis=0) # Sort rows (ChunkSize)
            pivot_data = pivot_data.sort_index(axis=1) # Sort columns (NumThreads)


            title = f"{scheduler_name} {metric_label} Heatmap - {workload_desc}"
            filename = f"heatmap_{metric.lower()}_{scheduler_name.lower().replace(' ', '_')}_W{workload_id}{file_suffix}.pdf"
            filepath = output_dir / filename

            # Choose appropriate colorscale (higher is better for speedup, lower for time)
            colorscale = 'Viridis' if show_speedup else 'Viridis_r' # _r reverses the scale

            fig = px.imshow(
                pivot_data,
                labels=dict(x="Number of Threads", y="Chunk Size", color=metric_label),
                # Explicitly set x and y to ensure correct order if pivot table is sparse
                x=pivot_data.columns.tolist(),
                y=pivot_data.index.tolist(),
                color_continuous_scale=colorscale,
                title=title,
                aspect="auto" # Adjust aspect ratio automatically
            )

            # Optional: Add text annotations to cells (can be cluttered)
            # for y_val in pivot_data.index:
            #     for x_val in pivot_data.columns:
            #         value = pivot_data.loc[y_val, x_val]
            #         text = f"{value:.2f}" if pd.notna(value) else ""
            #         fig.add_annotation(x=x_val, y=y_val, text=text, showarrow=False, font=dict(size=8))


            fig.update_layout(width=width, height=height)
            try:
                fig.write_image(filepath, format="pdf")
                print(f"  Saved: {filepath}")
            except Exception as e:
                print(f"  ERROR saving {filepath}: {e}")


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for Collatz implementations.")
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                        help=f"Path to the benchmark results CSV file (default: {DEFAULT_CSV_PATH})")
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f"Directory to save the plots (default: {DEFAULT_PLOT_DIR})")
    parser.add_argument('--fixed-chunk', type=int, default=DEFAULT_FIXED_CHUNK,
                        help=f"Fixed chunk size for Speedup/Time vs Threads plots (default: {DEFAULT_FIXED_CHUNK})")

    # Arguments to select plot types
    parser.add_argument('--speedup-vs-threads', action='store_true', help="Plot Speedup vs Number of Threads.")
    parser.add_argument('--time-vs-threads', action='store_true', help="Plot Execution Time vs Number of Threads (linear scale).")
    parser.add_argument('--time-vs-threads-log', action='store_true', help="Plot Execution Time vs Number of Threads (log scale).")
    parser.add_argument('--chunk-impact-speedup', action='store_true', help="Plot Speedup vs Chunk Size.")
    parser.add_argument('--chunk-impact-time', action='store_true', help="Plot Execution Time vs Chunk Size (linear scale).")
    parser.add_argument('--chunk-impact-time-log', action='store_true', help="Plot Execution Time vs Chunk Size (log scale).")
    parser.add_argument('--dynamic-chunks-comparison', action='store_true',
                        help="Plot Dynamic scheduler speedup vs threads for multiple chunk sizes.")
    parser.add_argument('--blockcyclic-chunks-comparison', action='store_true',
                        help="Plot Static Block-Cyclic speedup vs threads for multiple chunk sizes.")
    parser.add_argument('--scheduler-heatmaps', action='store_true',
                        help="Plot heatmaps of scheduler speedup across thread counts and chunk sizes.")
    parser.add_argument('--scheduler-heatmaps-time', action='store_true',
                        help="Plot heatmaps of execution time across thread counts and chunk sizes.")
    parser.add_argument('--all', action='store_true', help="Generate all supported plot types.")

    args = parser.parse_args()

    csv_file = Path(args.csv_path)
    plot_dir = Path(args.plot_dir)

    if not csv_file.is_file():
        print(f"Error: CSV file not found at {csv_file}")
        return

    # Create output directory
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved in: {plot_dir.resolve()}")

    # Load and Preprocess Data
    print(f"Loading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Preprocessing data...")
    # Convert 'N/A' or other non-numeric ChunkSize to NaN, then potentially handle later
    df['ChunkSize'] = pd.to_numeric(df['ChunkSize'], errors='coerce')
    # Convert times and speedup, coercing errors to NaN
    df['ExecutionTimeMs'] = pd.to_numeric(df['ExecutionTimeMs'], errors='coerce')
    df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
    # Convert NumThreads, filling potential NaNs after coercion (e.g., if empty) with 1 (for Sequential) before casting to int
    df['NumThreads'] = pd.to_numeric(df['NumThreads'], errors='coerce').fillna(1).astype(int)

    # Standard plot dimensions
    plot_width = 1000
    plot_height = 600

    # Determine which plots to generate based on flags
    generate_all = args.all
    plots_to_generate = {
        'speedup_vs_threads': generate_all or args.speedup_vs_threads,
        'time_vs_threads': generate_all or args.time_vs_threads,
        'time_vs_threads_log': generate_all or args.time_vs_threads_log,
        'chunk_impact_speedup': generate_all or args.chunk_impact_speedup,
        'chunk_impact_time': generate_all or args.chunk_impact_time,
        'chunk_impact_time_log': generate_all or args.chunk_impact_time_log,
        'dynamic_chunks_comparison': generate_all or args.dynamic_chunks_comparison,
        'blockcyclic_chunks_comparison': generate_all or args.blockcyclic_chunks_comparison,
        'scheduler_heatmaps': generate_all or args.scheduler_heatmaps,
        'scheduler_heatmaps_time': generate_all or args.scheduler_heatmaps_time,
    }

    if not any(plots_to_generate.values()):
        print("\nNo plot type selected. Use --all or specific flags like --speedup-vs-threads.")
        parser.print_help()
        return

    # Generate selected plots (pass copies of df to avoid unintended modifications)
    if plots_to_generate['speedup_vs_threads']:
        plot_speedup_vs_threads(df.copy(), plot_dir, args.fixed_chunk, width=plot_width, height=plot_height)

    if plots_to_generate['time_vs_threads']:
        plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, use_log_scale=False, width=plot_width, height=plot_height)

    if plots_to_generate['time_vs_threads_log']:
        plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_speedup']:
        plot_chunk_impact_speedup(df.copy(), plot_dir, width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_time']:
        plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=False, width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_time_log']:
        plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    if plots_to_generate['dynamic_chunks_comparison']:
        # Define chunk sizes likely present in the data for comparison
        dynamic_chunks = df[df['SchedulerName'] == 'Dynamic']['ChunkSize'].dropna().unique()
        plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic", chunk_sizes=sorted(dynamic_chunks), width=plot_width, height=plot_height)

    if plots_to_generate['blockcyclic_chunks_comparison']:
        blockcyclic_chunks = df[df['SchedulerName'] == 'Static Block-Cyclic']['ChunkSize'].dropna().unique()
        plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block-Cyclic", chunk_sizes=sorted(blockcyclic_chunks), width=plot_width, height=plot_height)

    if plots_to_generate['scheduler_heatmaps']:
        plot_scheduler_heatmaps(df.copy(), plot_dir, show_speedup=True, width=800, height=800) # Heatmaps often benefit from square aspect

    if plots_to_generate['scheduler_heatmaps_time']:
        plot_scheduler_heatmaps(df.copy(), plot_dir, show_speedup=False, width=800, height=800)

    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()
