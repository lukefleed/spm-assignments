#!/usr/bin/env python3
"""
Plotting scripts for Assignment 3 benchmarks.

Subcommands:
  --one_large        Heatmap matrix of speedup vs threads & block size (single large file)
  --many_small       Speedup vs threads plot (many small files) with Amdahl's ideal curve
  --many_large_sequential   Heatmap matrix for many large files (sequential dispatch)
  --many_large_parallel      Heatmap matrix for many large files (nested parallel)
  --many_large_parallel_right Heatmap matrix for many large files (controlled nesting)
  --all              Generate all plots

Usage:
  ./plot.py --one_large
  ./plot.py --many_small
  ./plot.py --many_large_sequential
  ./plot.py --many_large_parallel
  ./plot.py --many_large_parallel_right
  ./plot.py --all

Requires:
  pandas, plotly, kaleido
Install with: pip install pandas plotly kaleido
"""
import os
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_one_large(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_one_large.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    matrix = df.pivot(index='threads', columns='block_size', values='speedup')
    # convert block_size from bytes to MiB for axis
    matrix.columns = matrix.columns.astype(int) // (1024*1024)
    matrix.index = matrix.index.astype(str)
    fig = px.imshow(
        matrix,
        labels=dict(x="Block Size (MiB)", y="Threads", color="Speedup"),
        x=matrix.columns.astype(str),
        y=matrix.index,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Heatmap: Speedup vs Threads & Block Size (One Large File)',
        width=800,
        height=600,
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')
    out_dir = os.path.join(script_dir, 'results', 'plots', 'one_large')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_one_large.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"one_large plot saved to {out_pdf}")


def plot_many_small(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_small.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    df = df.sort_values('threads')
    # compute parallel fraction p from max threads data
    max_row = df.iloc[-1]
    T = max_row['threads']
    S = max_row['speedup']
    p = (1 - 1/S) / (1 - 1/T) if T > 1 else 0
    # ideal Amdahl curve
    ideal = df['threads'].apply(lambda t: 1/((1-p) + p/t))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['threads'], y=df['speedup'], mode='lines+markers', name='measured'
    ))
    fig.add_trace(go.Scatter(
        x=df['threads'], y=ideal, mode='lines', name='Amdahl'
    ))
    fig.update_layout(
        title='Strong Scaling Analysis: Many Small Files',
        xaxis_title='Number of Threads (p)', yaxis_title='Observed Speedup S(p)',
        width=800,
        height=600,
    )
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_small')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_many_small.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_small plot saved to {out_pdf}")


# Removed plot_block_speedup as it's redundant with heatmaps


def plot_many_large_sequential(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_large_sequential.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    # Create heatmap
    matrix = df.pivot(index='threads', columns='block_size', values='speedup')
    matrix.columns = matrix.columns.astype(int) // (1024*1024) # Convert block_size to MiB
    matrix.index = matrix.index.astype(str) # Keep threads as string categories
    fig = px.imshow(
        matrix,
        labels=dict(x="Block Size (MiB)", y="Inner Threads (p)", color="Speedup"),
        x=matrix.columns.astype(str),
        y=matrix.index,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Heatmap: Speedup vs Inner Threads & Block Size (Many Large Files - Sequential Dispatch)',
        width=800,
        height=600,
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_sequential')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_many_large_sequential.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_large_sequential plot saved to {out_pdf}")


def plot_many_large_parallel(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_large_parallel.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    # Create heatmap
    matrix = df.pivot(index='threads', columns='block_size', values='speedup')
    matrix.columns = matrix.columns.astype(int) // (1024*1024) # Convert block_size to MiB
    matrix.index = matrix.index.astype(str) # Keep threads as string categories
    fig = px.imshow(
        matrix,
        labels=dict(x="Block Size (MiB)", y="Threads per Level (p)", color="Speedup"),
        x=matrix.columns.astype(str),
        y=matrix.index,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Heatmap: Speedup vs Threads & Block Size (Many Large Files - Oversubscribed Nesting)',
        width=800,
        height=600,
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_parallel')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_many_large_parallel.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_large_parallel plot saved to {out_pdf}")


def plot_many_large_parallel_right(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_large_parallel_right.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    # Create heatmap similar to one_large
    matrix = df.pivot(index='threads', columns='block_size', values='speedup')
    matrix.columns = matrix.columns.astype(int) // (1024*1024)  # Convert block_size to MiB
    matrix.index = matrix.index.astype(str)  # Keep threads as string categories if needed

    fig = px.imshow(
        matrix,
        labels=dict(x="Block Size (MiB)", y="Total Threads (p)", color="Speedup"),
        x=matrix.columns.astype(str),
        y=matrix.index,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Heatmap: Speedup vs Threads & Block Size (Many Large Files - Controlled Nesting)',
        width=800,
        height=600,
    )

    # Remove annotations; just adjust layout
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')

    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_parallel_right')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_many_large_right.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_large_parallel_right plot saved to {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--one_large', action='store_true')
    group.add_argument('--many_small', action='store_true')
    # group.add_argument('--block_speedup', action='store_true') # Removed
    group.add_argument('--many_large_sequential', action='store_true')
    group.add_argument('--many_large_parallel', action='store_true')
    group.add_argument('--many_large_parallel_right', action='store_true')
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.one_large or args.all:
        plot_one_large(script_dir)
    if args.many_small or args.all:
        plot_many_small(script_dir)
    # if args.block_speedup or args.all: # Removed
    #     plot_block_speedup(script_dir)
    if args.many_large_sequential or args.all:
        plot_many_large_sequential(script_dir)
    if args.many_large_parallel or args.all:
        plot_many_large_parallel(script_dir) # Now plots heatmap
    if args.many_large_parallel_right or args.all:
        plot_many_large_parallel_right(script_dir)


if __name__ == '__main__':
    main()
