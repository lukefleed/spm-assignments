#!/usr/bin/env python3
"""
Plotting scripts for Assignment 3 benchmarks.

Subcommands:
  --one_large        Heatmap matrix of speedup vs threads & block size
  --many_small       Speedup vs threads plot (many small files) with Amdahl's ideal curve
  --block_speedup    Line plot of speedup vs block size at max threads
  --all              Generate all three plots

Usage:
  ./plot.py --one_large
  ./plot.py --many_small
  ./plot.py --block_speedup
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
    csv_file = os.path.join(script_dir, 'benchmark_matrix_results.csv')
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
        title='Speedup Matrix (one_large)'
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_autorange='reversed')
    out_dir = os.path.join(script_dir, 'results', 'plots', 'one_large')
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, 'speedup_matrix_one_large.pdf')
    fig.write_image(out_pdf, format='pdf')
    print(f"one_large plot saved to {out_pdf}")


def plot_many_small(script_dir):
    csv_file = os.path.join(script_dir, 'benchmark_many_small.csv')
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
        x=df['threads'], y=ideal, mode='lines', name='ideal Amdahl'
    ))
    fig.update_layout(
        title='Speedup vs Threads (many_small)',
        xaxis_title='Threads', yaxis_title='Speedup'
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
        title='Multiline Speedup vs Threads'
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
        fig.add_trace(go.Scatter(x=sub['threads'], y=ideal, mode='lines', name='ideal Amdahl'))
        fig.update_layout(
            title=f'Speedup vs Threads (Block {mib}MiB)',
            xaxis_title='Threads', yaxis_title='Speedup'
        )
        fig.write_image(os.path.join(out_dir, f'block_{mib}MiB_speedup.pdf'), format='pdf')
    print(f"block_speedup plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--one_large', action='store_true')
    group.add_argument('--many_small', action='store_true')
    group.add_argument('--block_speedup', action='store_true')
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.one_large or args.all:
        plot_one_large(script_dir)
    if args.many_small or args.all:
        plot_many_small(script_dir)
    if args.block_speedup or args.all:
        plot_block_speedup(script_dir)


if __name__ == '__main__':
    main()
