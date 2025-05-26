#!/usr/bin/env python3

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

def plot_single_node_performance(csv_filepath, output_dir):
    """
    Generates and saves performance plots as PDF for single-node results using Plotly.
    """
    if not os.path.exists(csv_filepath):
        print(f"Warning: Single-node results file not found: {csv_filepath}. Skipping single-node plots.", file=sys.stderr)
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}", file=sys.stderr)
        return

    if df.empty:
        print(f"Warning: {csv_filepath} is empty. No plots will be generated for single-node.", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)

    for (n_val, r_val), group in df.groupby(['N', 'R']):
        group = group.sort_values(by='T')

        baseline_time_series = group[group['T'] == 1]['mean_time_sec']
        if baseline_time_series.empty:
            print(f"Warning: No T=1 baseline data for N={n_val}, R={r_val}. Cannot calculate speedup/efficiency.", file=sys.stderr)
            time_t1 = None
        else:
            time_t1 = baseline_time_series.iloc[0]

        # --- Plot 1: Execution Time vs. Threads ---
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=group['T'],
            y=group['mean_time_sec'],
            mode='lines+markers',
            name='Mean Time',
            error_y=dict(type='data', array=group['std_dev_time_sec'], visible=True)
        ))
        fig_time.update_layout(
            title_text=f'Execution Time vs. Threads (N={n_val}, R={r_val})',
            xaxis_title='Number of FastFlow Threads (T)',
            yaxis_title='Mean Execution Time (seconds)',
            xaxis_type="log",
            xaxis_dtick=None, # Auto-ticks for log scale might be better
            xaxis_categoryorder='array', # Ensures T values are plotted as specified
            xaxis_categoryarray=sorted(group['T'].unique())
        )
        fig_time.update_xaxes(type='category') # Treat T as categorical for distinct ticks if not pure log scale
        fig_time.write_image(os.path.join(output_dir, f'ff_time_N{n_val}_R{r_val}.pdf'))


        if time_t1 is not None and time_t1 > 0:
            group['speedup'] = time_t1 / group['mean_time_sec']
            ideal_speedup = group['T']

            # --- Plot 2: Speedup vs. Threads ---
            fig_speedup = go.Figure()
            fig_speedup.add_trace(go.Scatter(
                x=group['T'],
                y=group['speedup'],
                mode='lines+markers',
                name='Actual Speedup'
            ))
            fig_speedup.add_trace(go.Scatter(
                x=group['T'],
                y=ideal_speedup,
                mode='lines',
                name='Ideal Speedup',
                line=dict(dash='dash')
            ))
            fig_speedup.update_layout(
                title_text=f'Speedup vs. Threads (N={n_val}, R={r_val})',
                xaxis_title='Number of FastFlow Threads (T)',
                yaxis_title='Speedup (S = T1 / TT)',
                xaxis_type="log",
                xaxis_dtick=None,
                xaxis_categoryorder='array',
                xaxis_categoryarray=sorted(group['T'].unique())
            )
            fig_speedup.update_xaxes(type='category')
            fig_speedup.write_image(os.path.join(output_dir, f'ff_speedup_N{n_val}_R{r_val}.pdf'))

            # --- Plot 3: Efficiency vs. Threads ---
            group['efficiency'] = group['speedup'] / group['T']
            fig_efficiency = go.Figure()
            fig_efficiency.add_trace(go.Scatter(
                x=group['T'],
                y=group['efficiency'],
                mode='lines+markers',
                name='Efficiency'
            ))
            fig_efficiency.update_layout(
                title_text=f'Efficiency vs. Threads (N={n_val}, R={r_val})',
                xaxis_title='Number of FastFlow Threads (T)',
                yaxis_title='Efficiency (E = S / T)',
                xaxis_type="log",
                xaxis_dtick=None,
                yaxis_range=[0, 1.1],
                xaxis_categoryorder='array',
                xaxis_categoryarray=sorted(group['T'].unique())
            )
            fig_efficiency.update_xaxes(type='category')
            fig_efficiency.write_image(os.path.join(output_dir, f'ff_efficiency_N{n_val}_R{r_val}.pdf'))

    print(f"Single-node PDF plots generated in {output_dir}")


def plot_hybrid_performance(csv_filepath, output_dir):
    """
    Generates and saves performance plots as PDF for hybrid results using Plotly.
    (Placeholder for strong/weak scaling)
    """
    if not os.path.exists(csv_filepath):
        print(f"Warning: Hybrid results file not found: {csv_filepath}. Skipping hybrid plots.", file=sys.stderr)
        return
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}", file=sys.stderr)
        return

    if df.empty:
        print(f"Warning: {csv_filepath} is empty. No plots will be generated for hybrid.", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Hybrid PDF plotting for {csv_filepath} is a placeholder. Implement strong/weak scaling plots.")
    # Example of what you might do for strong scaling:
    # Group by N, R, T_per_node (T in your CSV)
    # For each group, plot Time vs P (MPI processes)
    # Example for N=100M, R=64, T_per_node=4
    # specific_N = "100M" # Or however N is formatted
    # specific_R = 64
    # specific_T = 4
    # strong_scaling_df = df[(df['N'] == specific_N) & (df['R'] == specific_R) & (df['T'] == specific_T)]
    # if not strong_scaling_df.empty:
    #     strong_scaling_df = strong_scaling_df.sort_values(by='P')
    #     fig_strong = go.Figure()
    #     # ... add traces for time, speedup vs P ...
    #     fig_strong.update_layout(title_text=f'Strong Scaling (N={specific_N}, R={specific_R}, T_per_node={specific_T})')
    #     fig_strong.write_image(os.path.join(output_dir, f'hybrid_strong_scaling_N{specific_N}_R{specific_R}_T{specific_T}.pdf'))
    print(f"Hybrid PDF plots would be generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate performance plots as PDF from CSV benchmark results.")
    parser.add_argument("input_ff_csv", type=str, nargs='?', default="results_single_node.csv",
                        help="Path to the FastFlow single-node results CSV file (default: results_single_node.csv).")
    parser.add_argument("input_hybrid_csv", type=str, nargs='?', default="results_hybrid.csv",
                        help="Path to the Hybrid MPI+FastFlow results CSV file (default: results_hybrid.csv).")
    parser.add_argument("output_dir", type=str, default="build/plots",
                        help="Directory to save the generated PDF plots (default: build/plots).")

    args = parser.parse_args()

    ff_exists = os.path.exists(args.input_ff_csv)
    hybrid_exists = os.path.exists(args.input_hybrid_csv)

    if not ff_exists and not hybrid_exists:
        print("No CSV files found to plot. Please run benchmarks first.", file=sys.stderr)
        if args.input_ff_csv == "results_single_node.csv" and args.input_hybrid_csv == "results_hybrid.csv":
             print(f"Default files checked: '{args.input_ff_csv}', '{args.input_hybrid_csv}'", file=sys.stderr)
        else:
            if ff_exists:
                 print(f"Checked for (FF): '{args.input_ff_csv}' - Found", file=sys.stderr)
            else:
                 print(f"Checked for (FF): '{args.input_ff_csv}' - Not Found", file=sys.stderr)
            if hybrid_exists:
                 print(f"Checked for (Hybrid): '{args.input_hybrid_csv}' - Found", file=sys.stderr)
            else:
                 print(f"Checked for (Hybrid): '{args.input_hybrid_csv}' - Not Found", file=sys.stderr)
        sys.exit(1)

    if ff_exists:
        plot_single_node_performance(args.input_ff_csv, args.output_dir)
    else:
        print(f"Single-node results file '{args.input_ff_csv}' not found. Skipping single-node plots.", file=sys.stderr)


    if hybrid_exists:
        plot_hybrid_performance(args.input_hybrid_csv, args.output_dir)
    else:
        print(f"Hybrid results file '{args.input_hybrid_csv}' not found. Skipping hybrid plots.", file=sys.stderr)


if __name__ == "__main__":
    main()
