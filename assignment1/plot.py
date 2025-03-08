import pandas as pd
import plotly.graph_objects as go
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Read the CSV file
df = pd.read_csv("results.csv")

# Create figure
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
    xaxis_title='Vector Size',
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
