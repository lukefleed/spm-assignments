#!/usr/bin/env python3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
import os
from pathlib import Path

def plot_strong_scaling():
    """Plot strong scaling (fixed array size, varying threads)"""
    df = pd.read_csv('../results/performance_results.csv')

    # Filter only Parallel implementation
    df = df[df['Implementation'] == 'Parallel'].copy()

    # Get fixed parameters for title
    array_size = df['Data_Size'].iloc[0]
    payload = df['Payload_Size_Bytes'].iloc[0]

    fig = go.Figure()

    # Add speedup vs Sequential
    fig.add_trace(go.Scatter(x=df['Threads'], y=df['Speedup_vs_Sequential'],
                            mode='lines+markers', name='Speedup vs Sequential',
                            line=dict(color='blue')))

    # Add speedup vs std::sort
    fig.add_trace(go.Scatter(x=df['Threads'], y=df['Speedup_vs_StdSort'],
                            mode='lines+markers', name='Speedup vs std::sort',
                            line=dict(color='red')))

    fig.update_layout(
        title=f'Strong Scaling - Array Size: {array_size:,}, Payload: {payload} bytes',
        xaxis_title='Number of Threads',
        yaxis_title='Speedup',
        width=1000, height=800,
        # xaxis_type="log",  # Uncomment for log scale on x-axis
        # yaxis_type="log"   # Uncomment for log scale on y-axis
    )

    fig.write_image('plots/strong_scaling.pdf')
    print("Strong scaling plot saved to plots/strong_scaling.pdf")

def plot_payload_scaling():
    """Plot payload scaling (fixed threads and array, varying payload)"""
    df = pd.read_csv('../results/benchmark_payload_scaling_results.csv')

    # Filter only FastFlow Parallel implementation
    df = df[df['Implementation'] == 'FF_Parallel_MergeSort'].copy()

    # Get fixed parameters for title
    threads = df['Threads'].iloc[0]
    array_size = df['Data_Size'].iloc[0]

    fig = go.Figure()

    # Add speedup vs Sequential
    fig.add_trace(go.Scatter(x=df['Payload_Size_Bytes'], y=df['Speedup_vs_Sequential'],
                            mode='lines+markers', name='Speedup vs Sequential',
                            line=dict(color='blue')))

    # Add speedup vs std::sort
    fig.add_trace(go.Scatter(x=df['Payload_Size_Bytes'], y=df['Speedup_vs_StdSort'],
                            mode='lines+markers', name='Speedup vs std::sort',
                            line=dict(color='red')))

    fig.update_layout(
        title=f'Payload Scaling - Threads: {threads}, Array Size: {array_size:,}',
        xaxis_title='Payload Size (bytes)',
        yaxis_title='Speedup',
        width=1000, height=800,
        # xaxis_type="log",  # Uncomment for log scale on x-axis
        # yaxis_type="log"   # Uncomment for log scale on y-axis
    )

    fig.write_image('plots/payload_scaling.pdf')
    print("Payload scaling plot saved to plots/payload_scaling.pdf")

def plot_weak_scaling():
    """Plot weak scaling (fixed threads, varying array size)"""
    df = pd.read_csv('../results/benchmark_array_scaling_results.csv')

    # Filter only FastFlow Parallel implementation
    df = df[df['Implementation'] == 'FF_Parallel_MergeSort'].copy()

    # Get fixed parameters for title
    threads = df['Threads'].iloc[0]
    payload = df['Payload_Size_Bytes'].iloc[0]

    fig = go.Figure()

    # Add speedup vs Sequential
    fig.add_trace(go.Scatter(x=df['Data_Size'], y=df['Speedup_vs_Sequential'],
                            mode='lines+markers', name='Speedup vs Sequential',
                            line=dict(color='blue')))

    # Add speedup vs std::sort
    fig.add_trace(go.Scatter(x=df['Data_Size'], y=df['Speedup_vs_StdSort'],
                            mode='lines+markers', name='Speedup vs std::sort',
                            line=dict(color='red')))

    fig.update_layout(
        title=f'Weak Scaling - Fixed Threads: {threads}, Payload: {payload} bytes',
        xaxis_title='Array Size (elements)',
        yaxis_title='Speedup',
        width=1000, height=800,
        xaxis_type="log"  # Log scale for array size
    )

    fig.write_image('plots/weak_scaling.pdf')
    print("Weak scaling plot saved to plots/weak_scaling.pdf")

def plot_cluster_scaling():
    """Plot cluster/hybrid scaling (varying MPI nodes)"""
    df = pd.read_csv('../results/hybrid_performance_results.csv')

    # Filter only hybrid runs (not baseline)
    df = df[df['Test_Name'] == 'Hybrid_MPI_Parallel'].copy()

    # Use existing speedup column
    df['speedup'] = df['Parallel_Speedup']

    # Get fixed parameters for title
    threads = df['Parallel_Threads'].iloc[0]
    array_size = df['Data_Size'].iloc[0]
    payload = df['Payload_Size'].iloc[0]

    fig = px.line(df, x='MPI_Processes', y='speedup',
                  title=f'Hybrid MPI+Fastflow Scaling - Threads/Node: {threads}, Array Size: {array_size:,}, Payload: {payload} bytes',
                  labels={'MPI_Processes': 'Number of MPI Processes', 'speedup': 'Speedup vs Baseline'},
                  markers=True)

    fig.update_layout(
        width=1000, height=800,
        # xaxis_type="log",  # Uncomment for log scale on x-axis
        # yaxis_type="log"   # Uncomment for log scale on y-axis
    )
    fig.write_image('plots/cluster_scaling.pdf')
    print("Cluster scaling plot saved to plots/cluster_scaling.pdf")

def main():
    parser = argparse.ArgumentParser(description='Plot HPC scaling results')
    parser.add_argument('plot_type', choices=['strong', 'payload', 'weak', 'cluster', 'all'],
                       help='Type of plot to generate')

    args = parser.parse_args()

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    if args.plot_type == 'strong' or args.plot_type == 'all':
        plot_strong_scaling()

    if args.plot_type == 'payload' or args.plot_type == 'all':
        plot_payload_scaling()

    if args.plot_type == 'weak' or args.plot_type == 'all':
        plot_weak_scaling()

    if args.plot_type == 'cluster' or args.plot_type == 'all':
        plot_cluster_scaling()

if __name__ == '__main__':
    main()
