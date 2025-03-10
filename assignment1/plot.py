#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot Softmax benchmark results.')
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--parallel", action="store_true", help="Use parallel results")
group1.add_argument("--noparallel", action="store_true", help="Use non-parallel results")

group2 = parser.add_mutually_exclusive_group(required=True)
group2.add_argument("--avx512", action="store_true", help="Use AVX512 results")
group2.add_argument("--noavx512", action="store_true", help="Use non-AVX512 results")
args = parser.parse_args()

# Determine configuration strings based on input flags
config_parallel = "parallel" if args.parallel else "noparallel"
config_avx = "avx512" if args.avx512 else "noavx512"
config_tag = f"{config_parallel}_{config_avx}"

# Construct input CSV and speedup CSV file paths
results_csv = os.path.join("results", f"results_{config_tag}.csv")
speedup_csv = os.path.join("results", f"speedup_{config_tag}.csv")  # Not used below but available if needed

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Read the CSV file
df = pd.read_csv(results_csv)

# Create figure for all sizes
fig = go.Figure()

# Add traces for each implementation
fig.add_trace(go.Scatter(
    x=df['Size'],
    y=df['Plain'],
    mode='lines+markers',
    name='Plain Implementation'
))

fig.add_trace(go.Scatter(
    x=df['Size'],
    y=df['Auto'],
    mode='lines+markers',
    name='Auto Implementation'
))

fig.add_trace(go.Scatter(
    x=df['Size'],
    y=df['AVX'],
    mode='lines+markers',
    name='AVX Implementation'
))

# Update layout with labels and title
fig.update_layout(
    title=f"Softmax Implementation Performance Comparison ({config_parallel.capitalize()}, {'AVX512' if args.avx512 else 'No AVX512'})",
    xaxis_title='Vector Size (log scale)',
    yaxis_title='Execution Time (seconds, log scale)',
    legend_title='Implementation',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    xaxis_type='log',  # Using log scale for better visualization
    yaxis_type='log',  # Using log scale for better visualization
    template='plotly_white',
    width=900,
    height=600,
)

# Save the figure as PDF
pdf_out = os.path.join("images", f"softmax_{config_tag}.pdf")
fig.write_image(pdf_out)
print(f"Plot saved as {pdf_out}")

# ---------- Create Small Sizes Plot ----------

# Filter data for small sizes only (up to 8192)
df_small = df[df['Size'] <= 8192]

# Create figure for small sizes
fig_small = go.Figure()

# Add traces for each implementation (small sizes only)
fig_small.add_trace(go.Scatter(
    x=df_small['Size'],
    y=df_small['Plain'],
    mode='lines+markers',
    name='Plain Implementation'
))

fig_small.add_trace(go.Scatter(
    x=df_small['Size'],
    y=df_small['Auto'],
    mode='lines+markers',
    name='Auto Implementation'
))

fig_small.add_trace(go.Scatter(
    x=df_small['Size'],
    y=df_small['AVX'],
    mode='lines+markers',
    name='AVX Implementation'
))

# Update layout with labels and title for small sizes plot
fig_small.update_layout(
    title=f"Softmax Performance Comparison for Small Sizes ({config_parallel.capitalize()}, {'AVX512' if args.avx512 else 'No AVX512'})",
    xaxis_title='Vector Size',
    yaxis_title='Execution Time (seconds, log scale)',
    legend_title='Implementation',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    xaxis_type='log',  # Using log scale for better visualization
    yaxis_type='log',  # Using log scale for better visualization
    template='plotly_white',
    width=900,
    height=600,
)

# Save the small sizes figure as PDF
pdf_small_out = os.path.join("images", f"softmax_{config_tag}_small.pdf")
fig_small.write_image(pdf_small_out)
print(f"Small sizes plot saved as {pdf_small_out}")
