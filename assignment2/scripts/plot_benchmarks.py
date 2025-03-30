import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Per aggiungere linee orizzontali (baseline)
import os
import argparse
import numpy as np # Per NaN
from pathlib import Path

# --- Costanti ---
DEFAULT_CSV_PATH = "../results/performance_results.csv"
DEFAULT_PLOT_DIR = "../results/plots"
# Chunk size da usare per i grafici Speedup/Time vs Threads
# Scegli un valore rappresentativo o quello che ha dato risultati migliori
DEFAULT_FIXED_CHUNK = 64

# --- Funzioni di Plotting ---

def plot_speedup_vs_threads(df, plot_dir, fixed_chunk_size, file_suffix="", width=800, height=600):
    """Genera grafici Speedup vs Numero di Thread per ogni workload.

    Compara gli scheduler a un chunk_size fisso (per quelli che lo usano).
    Include Sequential e Static Cyclic per riferimento.
    """
    print(f"Plotting Speedup vs Threads (fixed chunk: {fixed_chunk_size})...")
    output_dir = plot_dir / "speedup_vs_threads"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filtra per chunk size fisso O per scheduler che non usano chunk size
    # Schedulers che non usano chunk size definito: Sequential, Static Cyclic
    schedulers_no_chunk = ["Sequential", "Static Cyclic"]

    # Filtra NaN in NumThreads e Speedup per sicurezza
    df_filtered = df.dropna(subset=['NumThreads', 'Speedup'])
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_plot = df_filtered[
        (df_filtered['ChunkSize'] == fixed_chunk_size) |
        (df_filtered['SchedulerName'].isin(schedulers_no_chunk))
    ].copy() # Usa .copy() per evitare SettingWithCopyWarning

    # Assicurati che NumThreads sia intero per l'asse X
    df_plot['NumThreads'] = df_plot['NumThreads'].astype(int)

    # Ordina per visualizzazione corretta delle linee
    df_plot.sort_values(by=['WorkloadID', 'SchedulerName', 'NumThreads'], inplace=True)

    # Colori specifici (opzionale, ma migliora la consistenza)
    color_map = {
        "Sequential": "black",
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Cyclic": px.colors.qualitative.Plotly[1],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for workload_id, group in df_plot.groupby('WorkloadID'):
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"speedup_vs_threads_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group,
                      x='NumThreads',
                      y='Speedup',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'NumThreads': 'Number of Threads', 'Speedup': 'Speedup (relative to Sequential)'},
                      color_discrete_map=color_map)

        # Aggiungi linea orizzontale per speedup = 1 (baseline)
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")

        # Assicura che l'asse X mostri tutti i valori dei thread come categorie se sono pochi, o lineare se molti
        # Converti NumThreads in stringa per forzare un asse categorico se ci sono pochi thread
        # Se hai molti thread (es. 1, 2, 4, 8, 12, 16, 32...), un asse lineare/log potrebbe essere meglio
        unique_threads = sorted(group['NumThreads'].unique())
        if len(unique_threads) < 8: # Soglia arbitraria per decidere se categorico o lineare
             fig.update_xaxes(type='category') # Tratta i thread come categorie discrete
        else:
             # Potresti volere un asse logaritmico se i thread scalano esponenzialmente
             # fig.update_xaxes(type='log')
             fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4) # Tick ogni 2 o 4

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}. Is Kaleido installed and working?")


def plot_time_vs_threads(df, plot_dir, fixed_chunk_size, use_log_scale=True, file_suffix="", width=800, height=600):
    """Genera grafici Execution Time vs Numero di Thread per ogni workload."""
    print(f"Plotting Execution Time vs Threads (fixed chunk: {fixed_chunk_size}, log_scale: {use_log_scale})...")
    output_dir = plot_dir / "time_vs_threads"
    output_dir.mkdir(parents=True, exist_ok=True)

    schedulers_no_chunk = ["Sequential", "Static Cyclic"]
    df_filtered = df.dropna(subset=['NumThreads', 'ExecutionTimeMs'])
    df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0] # Ignora errori o tempi <= 0
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_plot = df_filtered[
        (df_filtered['ChunkSize'] == fixed_chunk_size) |
        (df_filtered['SchedulerName'].isin(schedulers_no_chunk))
    ].copy()

    df_plot['NumThreads'] = df_plot['NumThreads'].astype(int)
    df_plot.sort_values(by=['WorkloadID', 'SchedulerName', 'NumThreads'], inplace=True)

    color_map = {
        "Sequential": "black",
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Cyclic": px.colors.qualitative.Plotly[1],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for workload_id, group in df_plot.groupby('WorkloadID'):
        workload_desc = group['WorkloadDescription'].iloc[0]
        y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
        title = f"Exec Time vs Threads - Workload: {workload_desc}<br>(Chunk Size = {fixed_chunk_size} for relevant schedulers)"
        filename = f"time_vs_threads_W{workload_id}{'_log' if use_log_scale else ''}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group,
                      x='NumThreads',
                      y='ExecutionTimeMs',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'NumThreads': 'Number of Threads', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=color_map)

        unique_threads = sorted(group['NumThreads'].unique())
        if len(unique_threads) < 8:
             fig.update_xaxes(type='category')
        else:
             fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4)

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")

def plot_chunk_impact_speedup(df, plot_dir, file_suffix="", width=800, height=600):
    """Genera grafici Speedup vs Chunk Size per ogni workload e num_threads."""
    print("Plotting Speedup vs Chunk Size impact...")
    output_dir = plot_dir / "chunk_impact_speedup"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Considera solo scheduler che USANO chunk size > 0 e dove abbiamo risultati validi
    # Static Block viene incluso perché è stato testato con chunk diversi, anche se potrebbe ignorarlo.
    # Potremmo escluderlo se confermato che l'implementazione lo ignora.
    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', 'Speedup'])
    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0] # Solo chunk size numerici > 0
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    # Ordina per visualizzazione corretta delle linee
    df_filtered.sort_values(by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'], inplace=True)

    # Colori specifici (solo per i rilevanti)
    color_map = {
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for (workload_id, num_threads), group in df_filtered.groupby(['WorkloadID', 'NumThreads']):
        # Salta num_threads = 1 se presente per errore
        if num_threads == 1:
             continue

        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"Speedup vs Chunk Size - Workload: {workload_desc}<br>(Threads = {num_threads})"
        filename = f"chunk_speedup_W{workload_id}_T{num_threads}{file_suffix}.pdf"
        filepath = output_dir / filename

        fig = px.line(group,
                      x='ChunkSize',
                      y='Speedup',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'ChunkSize': 'Chunk Size', 'Speedup': 'Speedup'},
                      color_discrete_map=color_map)

        # Usa asse X categorico perché abbiamo pochi valori discreti di chunk size
        fig.update_xaxes(type='category')

        fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")


def plot_chunk_impact_time(df, plot_dir, use_log_scale=True, file_suffix="", width=800, height=600):
    """Genera grafici Execution Time vs Chunk Size per ogni workload e num_threads."""
    print(f"Plotting Execution Time vs Chunk Size impact (log_scale: {use_log_scale})...")
    output_dir = plot_dir / "chunk_impact_time"
    output_dir.mkdir(parents=True, exist_ok=True)

    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', 'ExecutionTimeMs'])
    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0]
    df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    df_filtered.sort_values(by=['WorkloadID', 'NumThreads', 'SchedulerName', 'ChunkSize'], inplace=True)

    color_map = {
        "Static Block": px.colors.qualitative.Plotly[0],
        "Static Block-Cyclic": px.colors.qualitative.Plotly[2],
        "Dynamic": px.colors.qualitative.Plotly[3]
    }

    for (workload_id, num_threads), group in df_filtered.groupby(['WorkloadID', 'NumThreads']):
         if num_threads == 1:
             continue

         workload_desc = group['WorkloadDescription'].iloc[0]
         y_axis_label = 'Execution Time (ms)' + (' [Log Scale]' if use_log_scale else '')
         title = f"Exec Time vs Chunk Size - Workload: {workload_desc}<br>(Threads = {num_threads})"
         filename = f"chunk_time_W{workload_id}_T{num_threads}{'_log' if use_log_scale else ''}{file_suffix}.pdf"
         filepath = output_dir / filename

         fig = px.line(group,
                      x='ChunkSize',
                      y='ExecutionTimeMs',
                      color='SchedulerName',
                      markers=True,
                      title=title,
                      labels={'ChunkSize': 'Chunk Size', 'ExecutionTimeMs': y_axis_label},
                      log_y=use_log_scale,
                      color_discrete_map=color_map)

         fig.update_xaxes(type='category')
         fig.update_layout(legend_title_text='Scheduler', width=width, height=height)
         try:
             fig.write_image(filepath, format="pdf")
             print(f"  Saved: {filepath}")
         except Exception as e:
             print(f"  ERROR saving {filepath}: {e}")


# --- Funzione Principale ---

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for Collatz implementations.")
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                        help=f"Path to the benchmark results CSV file (default: {DEFAULT_CSV_PATH})")
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f"Directory to save the plots (default: {DEFAULT_PLOT_DIR})")
    parser.add_argument('--fixed-chunk', type=int, default=DEFAULT_FIXED_CHUNK,
                        help=f"Fixed chunk size to use for Speedup/Time vs Threads plots (default: {DEFAULT_FIXED_CHUNK})")

    # Argomenti per selezionare i tipi di plot
    parser.add_argument('--speedup-vs-threads', action='store_true', help="Plot Speedup vs Number of Threads.")
    parser.add_argument('--time-vs-threads', action='store_true', help="Plot Execution Time vs Number of Threads (linear scale).")
    parser.add_argument('--time-vs-threads-log', action='store_true', help="Plot Execution Time vs Number of Threads (log scale).")
    parser.add_argument('--chunk-impact-speedup', action='store_true', help="Plot Speedup vs Chunk Size.")
    parser.add_argument('--chunk-impact-time', action='store_true', help="Plot Execution Time vs Chunk Size (linear scale).")
    parser.add_argument('--chunk-impact-time-log', action='store_true', help="Plot Execution Time vs Chunk Size (log scale).")
    parser.add_argument('--all', action='store_true', help="Generate all plot types.")

    args = parser.parse_args()

    csv_file = Path(args.csv_path)
    plot_dir = Path(args.plot_dir)

    if not csv_file.is_file():
        print(f"Error: CSV file not found at {csv_file}")
        return

    # Crea directory di output
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved in: {plot_dir.resolve()}")

    # Carica e Preprocessa i Dati
    print(f"Loading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Preprocessing data...")
    # Converti 'N/A' in ChunkSize a NaN numerico
    df['ChunkSize'] = pd.to_numeric(df['ChunkSize'], errors='coerce') # 'coerce' trasforma non numerici in NaN
    # Converti tempi e speedup, gestendo errori
    df['ExecutionTimeMs'] = pd.to_numeric(df['ExecutionTimeMs'], errors='coerce')
    df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
    df['NumThreads'] = pd.to_numeric(df['NumThreads'], errors='coerce').fillna(0).astype(int) # Assumi 0 se NaN, poi int

    print("Data loaded and preprocessed:")
    print(df.info())
    # print(df.head()) # Descommenta per vedere le prime righe

    # Definisci dimensioni standard per tutti i plot
    plot_width = 1000
    plot_height = 600

    # Logica per decidere quali plot generare
    generate_all = args.all
    plots_to_generate = {
        'speedup_vs_threads': generate_all or args.speedup_vs_threads,
        'time_vs_threads': generate_all or args.time_vs_threads,
        'time_vs_threads_log': generate_all or args.time_vs_threads_log,
        'chunk_impact_speedup': generate_all or args.chunk_impact_speedup,
        'chunk_impact_time': generate_all or args.chunk_impact_time,
        'chunk_impact_time_log': generate_all or args.chunk_impact_time_log,
    }

    if not any(plots_to_generate.values()):
        print("\nNo plot type selected. Use --all or specific flags like --speedup-vs-threads.")
        parser.print_help()
        return

    # Genera i plot selezionati
    if plots_to_generate['speedup_vs_threads']:
        plot_speedup_vs_threads(df.copy(), plot_dir, args.fixed_chunk, width=plot_width, height=plot_height)

    if plots_to_generate['time_vs_threads']:
        plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, use_log_scale=False, width=plot_width, height=plot_height)

    if plots_to_generate['time_vs_threads_log']:
        plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_speedup']:
        plot_chunk_impact_speedup(df.copy(), plot_dir, width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_time']:
        plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=False, width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_time_log']:
        plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()
