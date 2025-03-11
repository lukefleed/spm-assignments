#!/usr/bin/env python3

import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
        xanchor="right",
        x=0.99
    ),
    template='plotly_white',
    width=900,
    height=600,
)

# Save the figure as PDF
pdf_out = os.path.join("images/thread_scaling", f"thread_scaling_{config_avx}.pdf")
fig.write_image(pdf_out)
print(f"Saved plot to {pdf_out}")
