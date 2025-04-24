#!/usr/bin/env python3
"""
Plotting scripts for Assignment 3 benchmarks.

Subcommands:
  --one_large        Heatmap matrix of speedup vs threads & block size
  --many_small       Speedup vs threads plot (many small files) with Amdahl's ideal curve
  --block_speedup    Line plot of speedup vs block size at max threads
  --many_large_sequential   Strong scaling for many large files (sequential dispatch)
  --many_large_parallel      Strong scaling for many large files (nested parallel)
  --all              Generate all plots

Usage:
  ./plot.py --one_large
  ./plot.py --many_small
  ./plot.py --block_speedup
  ./plot.py --many_large_sequential
  ./plot.py --many_large_parallel
  ./plot.py --all

Requires:
  pandas, plotly, kaleido
Install with: pip install pandas plotly kaleido
"""
import os
import sys
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
        title='Heatmap of Parallel Speedup | Single 512 MiB Dataset',
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
        title='Strong Scaling Analysis: 4000 files (1-50 KiB Each)',
        xaxis_title='Number of Threads (p)', yaxis_title='Observed Speedup S(p)',
        width=800,
        height=600,
    )
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_small')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_many_small.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_small plot saved to {out_pdf}")


def plot_block_speedup(script_dir):
    csv_file = os.path.join(script_dir, 'benchmark_matrix_results.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    # add block size in MiB column
    df['block_mib'] = df['block_size'] // (1024*1024)
    # select block sizes to plot
    selected = [1, 3, 6, 9, 12]
    df_sel = df[df['block_mib'].isin(selected)].sort_values(['block_mib', 'threads'])
    out_dir = os.path.join(script_dir, 'results', 'plots', 'block_speedup')
    ensure_dir(out_dir)
    # multiline plot (speedup vs threads for each selected block)
    fig_all = px.line(
        df_sel, x='threads', y='speedup', color='block_mib',
        markers=True, labels={'block_mib': 'Block Size (MiB)'},
        title='Strong Scaling across Thread Counts for Varying Block Sizes'
    )
    fig_all.write_image(os.path.join(out_dir, 'multiline_block_speedup.pdf'), format='pdf')
    # individual plots with Amdahl's ideal curve for each block size
    for mib in selected:
        sub = df[df['block_mib'] == mib].sort_values('threads')
        # compute parallel fraction p from max thread data
        max_row = sub[sub['threads'] == sub['threads'].max()].iloc[0]
        T = max_row['threads']; S = max_row['speedup']
        p = (1 - 1/S) / (1 - 1/T) if T > 1 else 0
        ideal = sub['threads'].apply(lambda t: 1/((1-p) + p/t))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub['threads'], y=sub['speedup'], mode='lines+markers', name='measured'))
        fig.add_trace(go.Scatter(x=sub['threads'], y=ideal, mode='lines', name='Amdahl'))
        fig.update_layout(
            title=f'Strong Scaling: Single 512 MiB File, Block Size = {mib} MiB',
            xaxis_title='Number of Threads (p)', yaxis_title='Observed Speedup S(p)',
            width=800,
            height=600,
        )
        fig.write_image(os.path.join(out_dir, f'block_{mib}MiB_speedup.pdf'), format='pdf')
    print(f"block_speedup plots saved to {out_dir}")


def plot_many_large_sequential(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_large_sequential.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    df = df.sort_values('threads')
    max_row = df.iloc[-1]
    T = max_row['threads']
    S = max_row['speedup']
    p = (1 - 1/S) / (1 - 1/T) if T > 1 else 0
    ideal = df['threads'].apply(lambda t: 1/((1-p) + p/t))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['threads'], y=df['speedup'], mode='lines+markers', name='measured'))
    fig.add_trace(go.Scatter(x=df['threads'], y=ideal, mode='lines', name='Amdahl'))
    fig.update_layout(
        title='Strong Scaling: Many Large Files Sequential Dispatch',
        xaxis_title='Number of Threads (p)', yaxis_title='Speedup',
        width=800, height=600
    )
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_sequential')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_many_large_sequential.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_large_sequential plot saved to {out_pdf}")


def plot_many_large_parallel(script_dir):
    csv_file = os.path.join(script_dir, 'results/data/benchmark_many_large_parallel.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    df = pd.read_csv(csv_file)
    df = df.sort_values('threads')
    max_row = df.iloc[-1]
    T = max_row['threads']
    S = max_row['speedup']
    p = (1 - 1/S) / (1 - 1/T) if T > 1 else 0
    ideal = df['threads'].apply(lambda t: 1/((1-p) + p/t))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['threads'], y=df['speedup'], mode='lines+markers', name='measured'))
    fig.add_trace(go.Scatter(x=df['threads'], y=ideal, mode='lines', name='Amdahl'))
    fig.update_layout(
        title='Strong Scaling: Many Large Files Nested Parallel',
        xaxis_title='Number of Threads (p)', yaxis_title='Speedup',
        width=800, height=600
    )
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_parallel')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_many_large_parallel.pdf')
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
    matrix.columns = matrix.columns.astype(int) // (1024*1024) # Convert block_size to MiB
    matrix.index = matrix.index.astype(str) # Keep threads as string categories if needed
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
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')
    out_dir = os.path.join(script_dir, 'results', 'plots', 'many_large_parallel_right')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_many_large_right.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"many_large_parallel_right plot saved to {out_pdf}")

    # Optional: Add individual line plots vs threads for selected block sizes if needed
    # (Similar to plot_block_speedup logic)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--one_large', action='store_true')
    group.add_argument('--many_small', action='store_true')
    group.add_argument('--block_speedup', action='store_true') # Might need update if source CSV changes
    group.add_argument('--many_large_sequential', action='store_true')
    group.add_argument('--many_large_parallel', action='store_true')
    group.add_argument('--many_large_parallel_right', action='store_true') # Add new flag
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.one_large or args.all:
        plot_one_large(script_dir)
    if args.many_small or args.all:
        plot_many_small(script_dir)
    # if args.block_speedup or args.all: # Check if this still makes sense
    #     plot_block_speedup(script_dir)
    if args.many_large_sequential or args.all:
        plot_many_large_sequential(script_dir)
    if args.many_large_parallel or args.all:
        # Assuming many_large_parallel also becomes a heatmap now
        plot_many_large_parallel(script_dir) # Make sure this function exists and plots heatmap
    if args.many_large_parallel_right or args.all: # Add new call
        plot_many_large_parallel_right(script_dir)


if __name__ == '__main__':
    main()
