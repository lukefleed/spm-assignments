import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Read the CSV file
df = pd.read_csv("results/results_parallel_noaxv521.csv")
# df_speedup = pd.read_csv("speedup_noparallel_axv521.csv")

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
    title='Softmax Implementation Performance Comparison (Parallel, No AXV521)',
    xaxis_title='Vector Size (log scale) ',
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
fig.write_image("images/softmax_parallel_noaxv521.pdf")
print("Plot saved as images/softmax_parallel_noaxv521.pdf")

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
    title='Softmax Performance Comparison for Small Sizes (Parallel, No AXV521)',
    xaxis_title='Vector Size ',
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
fig_small.write_image("images/softmax_parallel_noaxv521_small.pdf")
print("Small sizes plot saved as images/softmax_parallel_noaxv521_small.pdf")

# # ---------- Create Histogram Plot for Speedups with Extended Non-Power-of-2 Categories ----------

# # Helper function to determine size category (for tiny, small, medium, large)
# def size_category(k):
#     if k <= 64:
#         return "tiny"
#     elif k <= 1024:
#         return "small"
#     elif k <= 16384:
#         return "medium"
#     else:
#         return "large"

# # Helper function to check if a number is a power of 2
# def is_power_of_two(n):
#     return n > 0 and (n & (n - 1)) == 0

# # Helper function to classify non-power-of-2 sizes into small or large
# def non_pow2_category(k):
#     if is_power_of_two(k):
#         return None
#     else:
#         return "small non-pow2" if k <= 16384 else "large non-pow2"

# # Create new columns for grouping
# df_speedup['size_category'] = df_speedup['Size'].apply(size_category)
# df_speedup['pow2_category'] = df_speedup['Size'].apply(lambda x: "pow2" if is_power_of_two(x) else None)
# df_speedup['non_pow2_category'] = df_speedup['Size'].apply(non_pow2_category)

# # Define groups to aggregate:
# # Four size intervals, power-of-2, and non-power-of-2 split into small and large
# groups = ['tiny', 'small', 'medium', 'large', 'pow2', 'small non-pow2', 'large non-pow2']
# avg_auto_speedup = []
# avg_avx_speedup = []

# for group in groups:
#     if group in ['tiny', 'small', 'medium', 'large']:
#         subset = df_speedup[df_speedup['size_category'] == group]
#     elif group == 'pow2':
#         subset = df_speedup[df_speedup['pow2_category'] == 'pow2']
#     elif group in ['small non-pow2', 'large non-pow2']:
#         subset = df_speedup[df_speedup['non_pow2_category'] == group]
#     else:
#         subset = pd.DataFrame()  # fallback, shouldn't happen
#     avg_auto_speedup.append(subset['Auto_Speedup'].mean())
#     avg_avx_speedup.append(subset['AVX_Speedup'].mean())

# # Create grouped bar chart with the updated groups
# fig_speedup = go.Figure(data=[
#     go.Bar(
#         name='Auto Speedup',
#         x=groups,
#         y=avg_auto_speedup,
#         text=[f"{val:.1f}x" if pd.notnull(val) else "" for val in avg_auto_speedup],
#         textposition='auto'
#     ),
#     go.Bar(
#         name='AVX Speedup',
#         x=groups,
#         y=avg_avx_speedup,
#         text=[f"{val:.1f}x" if pd.notnull(val) else "" for val in avg_avx_speedup],
#         textposition='auto'
#     )
# ])
# fig_speedup.update_layout(
#     barmode='group',
#     title='Average Speedup relative to Plain Implementation (No Parallel, No AXV521)',
#     legend=dict(
#         yanchor="top",
#         y=0.99,
#         xanchor="left",
#         x=0.01
#     ),
#     xaxis_title='Size of the input',
#     yaxis_title='Average Speedup',
#         template='plotly_white',
#     width=1000,
#     height=600,
# )

# # Save the updated speedup histogram as PDF
# fig_speedup.write_image("images/softmax_parallel_axv521_speedup.pdf")
# print("Speedup plot saved as images/softmax_noparallel_axv521_speedup.pdf")
