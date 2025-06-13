#!/usr/bin/env python3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
import os

def plot_strong_scaling_analysis():
    """Plots strong scaling analysis for the single-node parallel implementation."""
    try:
        df = pd.read_csv('../results/performance_results.csv')
    except FileNotFoundError:
        print("Warning: '../results/performance_results.csv' not found. Skipping strong scaling plot.")
        return

    df = df[df['Implementation'] == 'Parallel'].copy()
    if df.empty:
        print("Warning: No 'Parallel' implementation data in 'performance_results.csv'.")
        return

    array_size = df['Data_Size'].iloc[0]
    payload = df['Payload_Size_Bytes'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Threads'], y=df['Speedup_vs_Sequential'],
                            mode='lines+markers', name='Speedup vs. Sequential'))
    fig.add_trace(go.Scatter(x=df['Threads'], y=df['Speedup_vs_StdSort'],
                            mode='lines+markers', name='Speedup vs. std::sort'))

    fig.update_layout(
        title=f'Single-Node Strong Scaling Analysis<br><sup>N = {array_size:,}, Payload = {payload} bytes</sup>',
        xaxis_title='Number of Threads (T)',
        yaxis_title='Speedup',
        legend_title_text='Baseline Comparison',
        width=1000, height=800
    )

    fig.write_image('plots/strong_scaling.pdf')
    print("Generated 'plots/strong_scaling.pdf'")

def plot_payload_sensitivity_analysis():
    """Plots the impact of payload size on performance."""
    try:
        df = pd.read_csv('../results/benchmark_payload_scaling_results.csv')
    except FileNotFoundError:
        print("Warning: '../results/benchmark_payload_scaling_results.csv' not found. Skipping payload scaling plot.")
        return

    df = df[df['Implementation'] == 'FF_Parallel_MergeSort'].copy()
    if df.empty:
        print("Warning: No 'FF_Parallel_MergeSort' data in 'benchmark_payload_scaling_results.csv'.")
        return

    threads = df['Threads'].iloc[0]
    array_size = df['Data_Size'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Payload_Size_Bytes'], y=df['Execution_Time_ms'],
                            mode='lines+markers', name='Execution Time'))

    fig.update_layout(
        title=f'Payload Size Sensitivity Analysis<br><sup>N = {array_size:,}, T = {threads}</sup>',
        xaxis_title='Payload Size (bytes)',
        yaxis_title='Execution Time (ms)',
        width=1000, height=800
    )

    fig.write_image('plots/payload_scaling.pdf')
    print("Generated 'plots/payload_scaling.pdf'")

def plot_problem_size_scaling_analysis():
    """Plots performance scaling with problem size for a fixed number of threads."""
    try:
        df = pd.read_csv('../results/benchmark_array_scaling_results.csv')
    except FileNotFoundError:
        print("Warning: '../results/benchmark_array_scaling_results.csv' not found. Skipping problem size scaling plot.")
        return

    df = df[df['Implementation'] == 'FF_Parallel_MergeSort'].copy()
    if df.empty:
        print("Warning: No 'FF_Parallel_MergeSort' data in 'benchmark_array_scaling_results.csv'.")
        return

    threads = df['Threads'].iloc[0]
    payload = df['Payload_Size_Bytes'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Data_Size'], y=df['Speedup_vs_Sequential'],
                            mode='lines+markers', name='Speedup vs. Sequential'))
    fig.add_trace(go.Scatter(x=df['Data_Size'], y=df['Speedup_vs_StdSort'],
                            mode='lines+markers', name='Speedup vs. std::sort'))

    fig.update_layout(
        title=f'Single-Node Problem Size Scaling Analysis<br><sup>T = {threads}, Payload = {payload} bytes</sup>',
        xaxis_title='Problem Size (N)',
        yaxis_title='Speedup',
        xaxis_type="log",
        legend_title_text='Baseline Comparison',
        width=1000, height=800
    )

    fig.write_image('plots/weak_scaling.pdf')
    print("Generated 'plots/weak_scaling.pdf'")

def plot_cluster_strong_scaling_analysis():
    """Plots strong scaling for the hybrid MPI+FastFlow implementation."""
    try:
        df = pd.read_csv('../results/hybrid_performance_results.csv')
    except FileNotFoundError:
        print("Warning: '../results/hybrid_performance_results.csv' not found. Skipping cluster strong scaling plot.")
        return

    # Filter only hybrid runs, exclude baseline
    df_hybrid = df[df['Test_Name'] == 'Hybrid_MPI_Parallel'].copy()
    # Get baseline time
    baseline_time_series = df[df['Test_Name'] == 'Parallel_Baseline']['Total_Time_ms']
    if baseline_time_series.empty:
        print("Warning: Baseline data not found for cluster strong scaling.")
        return
    baseline_time = baseline_time_series.iloc[0]

    if df_hybrid.empty:
        print("Warning: No 'Hybrid_MPI_Parallel' data in 'hybrid_performance_results.csv'.")
        return

    df_hybrid['Speedup'] = baseline_time / df_hybrid['Total_Time_ms']

    threads_per_node = df_hybrid['Parallel_Threads'].iloc[0]
    array_size = df_hybrid['Data_Size'].iloc[0]
    payload = df_hybrid['Payload_Size'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hybrid['MPI_Processes'], y=df_hybrid['Speedup'],
                            mode='lines+markers', name='Measured Speedup'))

    fig.update_layout(
        title=f'Cluster Strong Scaling Analysis (MPI)<br><sup>N = {array_size:,}, T/Node = {threads_per_node}, Payload = {payload} bytes</sup>',
        xaxis_title='Number of MPI Processes (P)',
        yaxis_title='Speedup (vs. 1-Node Parallel)',
        width=1000, height=800
    )

    fig.write_image('plots/cluster_scaling.pdf')
    print("Generated 'plots/cluster_scaling.pdf'")

def plot_cluster_weak_scaling_analysis():
    """Plots weak scaling efficiency for the hybrid MPI+FastFlow implementation."""
    try:
        df = pd.read_csv('../results/cluster_weak_scaling_results.csv')
    except FileNotFoundError:
        print("Warning: '../results/cluster_weak_scaling_results.csv' not found. Skipping cluster weak scaling plot.")
        return

    if df.empty:
        print("Warning: 'cluster_weak_scaling_results.csv' is empty.")
        return

    # Baseline time is for the first entry (smallest number of processes)
    baseline_time = df['Total_Time_ms'].iloc[0]

    # Calculate weak scaling efficiency: E = T_baseline / T_p
    df['Efficiency'] = baseline_time / df['Total_Time_ms']

    threads_per_node = df['Parallel_Threads'].iloc[0]
    payload = df['Payload_Size'].iloc[0]
    records_per_node = df['Data_Size'].iloc[0] // df['MPI_Processes'].iloc[0]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['MPI_Processes'], y=df['Efficiency'],
                            mode='lines+markers', name='Measured Efficiency'))
    # Add ideal efficiency line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Ideal Efficiency")

    fig.update_layout(
        title=f'Cluster Weak Scaling Analysis (MPI)<br><sup>N/P = {records_per_node:,}, T/Node = {threads_per_node}, Payload = {payload} bytes</sup>',
        xaxis_title='Number of MPI Processes (P)',
        yaxis_title='Weak Scaling Efficiency (T_baseline / T_p)',
        yaxis=dict(range=[0, df['Efficiency'].max() * 1.1]),
        width=1000, height=800
    )

    fig.write_image('plots/cluster_weak_scaling.pdf')
    print("Generated 'plots/cluster_weak_scaling.pdf'")


def main():
    parser = argparse.ArgumentParser(description='Plot HPC scaling results and analyses.')
    parser.add_argument('plot_type',
                       choices=['strong', 'payload', 'problem_size', 'cluster_strong', 'cluster_weak', 'all'],
                       help='Type of plot to generate.')

    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    plot_map = {
        'strong': plot_strong_scaling_analysis,
        'payload': plot_payload_sensitivity_analysis,
        'problem_size': plot_problem_size_scaling_analysis,
        'cluster_strong': plot_cluster_strong_scaling_analysis,
        'cluster_weak': plot_cluster_weak_scaling_analysis
    }

    if args.plot_type == 'all':
        for plot_func in plot_map.values():
            try:
                plot_func()
            except Exception as e:
                print(f"Error generating plot for {plot_func.__name__}: {e}")
    elif args.plot_type in plot_map:
        try:
            plot_map[args.plot_type]()
        except Exception as e:
            print(f"Error generating plot for {args.plot_type}: {e}")

if __name__ == '__main__':
    main()
