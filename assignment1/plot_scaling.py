#!/usr/bin/env python3

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Configure Plotly to not use MathJax by default
pio.kaleido.scope.mathjax = None

# Hardcoded configuration - always use AVX512
config_avx = "avx512"

# Construct input CSV file path
results_csv = os.path.join("results/thread_scaling", f"thread_scaling_{config_avx}.csv")

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)
os.makedirs("images/thread_scaling", exist_ok=True)

# Read the CSV file
df = pd.read_csv(results_csv)

# Create figure
fig = go.Figure()

# Add traces for each implementation
fig.add_trace(go.Scatter(
    x=df['Threads'],
    y=df['Auto'],
    mode='lines+markers',
    name='Auto-vectorized Implementation'
))

fig.add_trace(go.Scatter(
    x=df['Threads'],
    y=df['AVX'],
    mode='lines+markers',
    name='AVX Implementation'
))

# Update layout with labels and title
fig.update_layout(
    title=f"Thread Scaling Performance (K=2^30, AVX512, Parallel)",
    xaxis_title='Number of Threads',
    yaxis_title='Execution Time (seconds, log scale)',
    # xaxis_type='log',
    yaxis_type='log',
    legend_title='Implementation',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    template='plotly_white',
    width=900,
    height=600,
)

# Save the first plot
pdf_out = os.path.join("images/thread_scaling", f"thread_scaling_{config_avx}.pdf")
pio.write_image(fig, pdf_out, format='pdf')  # Explicitly specify format
print(f"Saved plot to {pdf_out}")


fig_speedup = go.Figure()

# Calculate speedups for both implementations
# Base time is the time with 1 thread
auto_base = df.loc[df['Threads'] == 1, 'Auto'].values[0]
avx_base = df.loc[df['Threads'] == 1, 'AVX'].values[0]

df['Auto_Speedup'] = auto_base / df['Auto']
df['AVX_Speedup'] = avx_base / df['AVX']

# Add traces for actual speedups
fig_speedup.add_trace(go.Scatter(
    x=df['Threads'],
    y=df['Auto_Speedup'],
    mode='lines+markers',
    name='Auto-vectorized Implementation'
))

fig_speedup.add_trace(go.Scatter(
    x=df['Threads'],
    y=df['AVX_Speedup'],
    mode='lines+markers',
    name='AVX Implementation'
))

# Add Amdahl's Law curve
# Estimate parallel fraction using max observed speedup
# Formula: max_speedup = 1 / ((1-p) + p/N)
# where p is the parallel fraction
# Solving for p: p = (1/max_speedup - 1) / (1/max_threads - 1)
max_threads = df['Threads'].max()
max_speedup_auto = df['Auto_Speedup'].max()
max_speedup_avx = df['AVX_Speedup'].max()

# Estimate parallel fraction for both implementations
p_auto = (1/max_speedup_auto - 1) / (1/max_threads - 1)
p_avx = (1/max_speedup_avx - 1) / (1/max_threads - 1)

# Use the average parallel fraction for Amdahl's Law curve
p = (p_auto + p_avx) / 2
p = min(max(p, 0.5), 0.99)  # Constrain to reasonable values

threads = np.linspace(1, max_threads, 100)
amdahl_speedup = 1 / ((1 - p) + p / threads)

fig_speedup.add_trace(go.Scatter(
    x=threads,
    y=amdahl_speedup,
    mode='lines',
    line=dict(dash='dash'),
    name=f"Amdahl's Law"
))

# Add ideal speedup line
fig_speedup.add_trace(go.Scatter(
    x=[1, max_threads],
    y=[1, max_threads],
    mode='lines',
    line=dict(dash='dot'),
    name='Ideal Linear Speedup'
))

# Update layout with labels and title
fig_speedup.update_layout(
    title=f"Thread Scaling Speedup (K=2^30, AVX512, Parallel)",
    xaxis_title='Number of Threads',
    yaxis_title='Speedup',
    legend_title='Implementation',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    template='plotly_white',
    width=900,
    height=600,
)

# Save the speedup plot
pdf_out_speedup = os.path.join("images/thread_scaling", f"thread_scaling_speedup_{config_avx}.pdf")
pio.write_image(fig_speedup, pdf_out_speedup, format='pdf')  # Explicitly specify format
print(f"Saved speedup plot to {pdf_out_speedup}")
