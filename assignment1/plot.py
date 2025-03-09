import pandas as pd
import plotly.graph_objects as go
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Read the CSV file
df = pd.read_csv("results.csv")

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
    title='Softmax Implementation Performance Comparison',
    xaxis_title='Vector Size (log scale)',
    yaxis_title='Execution Time (seconds)',
    legend_title='Implementation',
    xaxis_type='log',  # Using log scale for better visualization
    template='plotly_white',
    width=1000,
    height=600,
)

# Save the figure as SVG
fig.write_image("images/softmax_performance.svg")
print("Plot saved as images/softmax_performance.svg")

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
    title='Softmax Performance Comparison (Small Sizes ≤ 8192)',
    xaxis_title='Vector Size (log scale)',
    yaxis_title='Execution Time (seconds)',
    legend_title='Implementation',
    xaxis_type='log',  # Using log scale for better visualization
    template='plotly_white',
    width=1000,
    height=600,
)

# Save the small sizes figure as SVG
fig_small.write_image("images/softmax_performance_small.svg")
print("Small sizes plot saved as images/softmax_performance_small.svg")
