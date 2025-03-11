#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot Softmax benchmark results.')
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--parallel", action="store_true", help="Use parallel results")
group1.add_argument("--noparallel", action="store_true", help="Use non-parallel results")

group2 = parser.add_mutually_exclusive_group(required=True)
group2.add_argument("--avx512", action="store_true", help="Use AVX512 results")
group2.add_argument("--noavx512", action="store_true", help="Use non-AVX512 results")

# Add stability analysis option
parser.add_argument("--stability", action="store_true", help="Generate numerical stability analysis plots")

args = parser.parse_args()

# Determine configuration strings based on input flags
config_parallel = "parallel" if args.parallel else "noparallel"
config_avx = "avx512" if args.avx512 else "noavx512"
config_tag = f"{config_parallel}_{config_avx}"

# Construct input CSV and speedup CSV file paths
results_csv = os.path.join("results", f"results_{config_tag}.csv")
speedup_csv = os.path.join("results", f"speedup_{config_tag}.csv")
stability_csv = os.path.join("results", f"stability_{config_tag}.csv")

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
    marker=dict(size=2),
    name='Plain Implementation'
))

fig.add_trace(go.Scatter(
    x=df['Size'],
    y=df['Auto'],
    mode='lines+markers',
    marker=dict(size=2),
    name='Auto Implementation'
))

fig.add_trace(go.Scatter(
    x=df['Size'],
    y=df['AVX'],
    mode='lines+markers',
    marker=dict(size=2),
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
    marker=dict(size=2),
    name='Plain Implementation'
))

fig_small.add_trace(go.Scatter(
    x=df_small['Size'],
    y=df_small['Auto'],
    mode='lines+markers',
    marker=dict(size=2),
    name='Auto Implementation'
))

fig_small.add_trace(go.Scatter(
    x=df_small['Size'],
    y=df_small['AVX'],
    mode='lines+markers',
    marker=dict(size=2),
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

# ---------- Create Speedup Bar Chart ----------

# Define the sizes of interest as powers of 2
selected_sizes = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
# Read the speedup CSV file (constructed at the top with same naming logic)
df_speedup = pd.read_csv(speedup_csv)

# Filter rows for the selected sizes
df_selected = df_speedup[df_speedup['Size'].isin(selected_sizes)].copy()

# Ensure the sizes appear in the desired order
df_selected['Size'] = pd.Categorical(df_selected['Size'], categories=selected_sizes, ordered=True)
df_selected.sort_values('Size', inplace=True)

# Create grouped bar chart for speedup comparison
fig_speedup = go.Figure()

# Add Auto speedup bars
fig_speedup.add_trace(go.Bar(
    x=[str(s) for s in df_selected['Size']],
    y=df_selected['Auto_Speedup'],
    name='Auto Speedup',
    text=[f"{val:.1f}x" for val in df_selected['Auto_Speedup']],
    textposition='auto'
))

# Add AVX speedup bars
fig_speedup.add_trace(go.Bar(
    x=[str(s) for s in df_selected['Size']],
    y=df_selected['AVX_Speedup'],
    name='AVX Speedup',
    text=[f"{val:.1f}x" for val in df_selected['AVX_Speedup']],
    textposition='auto'
))

# Update layout with labels and title
fig_speedup.update_layout(
    title=f"Speedup Comparison ({config_parallel.capitalize()}, {'AVX512' if args.avx512 else 'No AVX512'})",
    xaxis_title='Vector Size',
    yaxis_title='Speedup Factor (higher is better)',
    barmode='group',
    template='plotly_white',
    width=900,
    height=600,
)

# Save the speedup bar chart as PDF
pdf_speedup_out = os.path.join("images", f"softmax_speedup_{config_tag}.pdf")
fig_speedup.write_image(pdf_speedup_out)
print(f"Speedup bar chart saved as {pdf_speedup_out}")

if args.stability:
    # Load stability data
    try:
        stability_data = pd.read_csv(stability_csv)

        # Create a single plot for sum values
        fig_stability = go.Figure()

        # Plot sum values for each implementation
        fig_stability.add_trace(
            go.Scatter(x=stability_data['Size'], y=stability_data['PlainSum'],
                      mode='lines+markers', name='Plain Implementation',
                      marker=dict(size=6))
        )
        fig_stability.add_trace(
            go.Scatter(x=stability_data['Size'], y=stability_data['AutoSum'],
                      mode='lines+markers', name='Auto Implementation',
                      marker=dict(size=6))
        )
        fig_stability.add_trace(
            go.Scatter(x=stability_data['Size'], y=stability_data['AVXSum'],
                      mode='lines+markers', name='AVX Implementation',
                      marker=dict(size=6))
        )

        # Update layout
        fig_stability.update_layout(
            title=f"Numerical Stability: Sum Analysis ({config_parallel.capitalize()}, {'AVX512' if args.avx512 else 'No AVX512'})",
            xaxis_title='Vector Size',
            yaxis_title='Sum of Softmax Output (closer to 1.0 is better)',
            # yaxis_type='log',
            # xaxis_type='log',
            template='plotly_white',
            width=900,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            # Set y-axis range to make 1.0 exactly in the middle
            yaxis=dict(
                range=[0.99, 1.01]  # Range centered at 1.0
            )
        )

        # Add a reference line at y=1.0 (ideal sum for softmax)
        fig_stability.add_shape(
            type="line",
            x0=stability_data['Size'].min(),
            y0=1.0,
            x1=stability_data['Size'].max(),
            y1=1.0,
            line=dict(
                color="black",
                width=2,
                dash="dash",
            ),
        )

        # Save figure
        stability_pdf_out = os.path.join("images", f"stability_{config_tag}.pdf")
        fig_stability.write_image(stability_pdf_out)
        print(f"Sum stability plot saved as {stability_pdf_out}")

    except FileNotFoundError:
        print(f"Warning: Stability data file {stability_csv} not found. Skipping stability analysis.")
    except Exception as e:
        print(f"Error generating stability plots: {e}")
