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

def plot_speedup_vs_threads(df, plot_dir, fixed_chunk_size, file_suffix="", width=1000, height=600):
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
        unique_threads = sorted(group['NumThreads'].unique())
        if len(unique_threads) < 8: # Soglia arbitraria per decidere se categorico o lineare
             fig.update_xaxes(type='category') # Tratta i thread come categorie discrete
        else:
             fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4)

        # Stima la frazione sequenziale analizzando i dati di speedup
        # Prendiamo il miglior speedup osservato con il massimo numero di thread e calcoliamo
        # una stima della frazione sequenziale usando la formula inversa di Amdahl
        max_threads = max(unique_threads)
        if max_threads > 1:  # Serve più di un thread per avere speedup
            # Trova il miglior speedup per il massimo numero di thread
            max_thread_data = group[group['NumThreads'] == max_threads]
            if not max_thread_data.empty:
                best_speedup = max_thread_data['Speedup'].max()
                if best_speedup > 1:  # Se c'è effettivamente speedup
                    # Stima della frazione sequenziale usando la formula inversa di Amdahl
                    # S(n) = 1 / (s + (1-s)/n) => s = (1 - S(n)/n) / (1 - 1/n)
                    s = (1 - best_speedup/max_threads) / (1 - 1/max_threads)
                    s = max(0.01, min(0.99, s))  # Limita a valori ragionevoli tra 1% e 50%

                    # Aggiungi curva di Amdahl per la frazione sequenziale stimata
                    amdahl_x = list(range(1, max(unique_threads) + 1)) if len(unique_threads) < 8 else np.linspace(1, max(unique_threads), 100)
                    amdahl_y = [1 / (s + (1-s)/n) for n in amdahl_x]

                    fig.add_trace(go.Scatter(
                        x=amdahl_x,
                        y=amdahl_y,
                        mode='lines',
                        line=dict(color='red', dash='dash', width=1.5),
                        name=f"Amdahl's Law (s={s:.2f})",
                        showlegend=True
                    ))

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


def plot_scheduler_chunk_comparison(df, plot_dir, scheduler_name, chunk_sizes=[16, 32, 64, 96, 128, 256], file_suffix="", width=1000, height=600):
    """Genera grafici Speedup vs Threads per un singolo scheduler con linee multiple per chunk size."""
    print(f"Plotting {scheduler_name} Speedup vs Threads for different chunk sizes...")
    output_dir = plot_dir / f"{scheduler_name.lower().replace(' ', '_')}_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filtra il dataframe per lo scheduler specificato
    df_scheduler = df[df['SchedulerName'] == scheduler_name].copy()

    # Filtra per i chunk size specificati
    df_filtered = df_scheduler[df_scheduler['ChunkSize'].isin(chunk_sizes)]

    # Converti a int e ordina
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered.sort_values(by=['WorkloadID', 'ChunkSize', 'NumThreads'], inplace=True)

    # Genera colori distinti per i chunk size
    # Usa una colorscale sequenziale per rappresentare la grandezza incrementale
    color_sequence = px.colors.sequential.Viridis
    n_chunks = len(chunk_sizes)
    color_map = {chunk: color_sequence[i * len(color_sequence) // n_chunks]
                for i, chunk in enumerate(chunk_sizes)}

    for workload_id, group in df_filtered.groupby('WorkloadID'):
        workload_desc = group['WorkloadDescription'].iloc[0]
        title = f"{scheduler_name} Speedup vs Threads - Workload: {workload_desc}<br>(Comparison of different chunk sizes)"
        filename = f"{scheduler_name.lower().replace(' ', '_')}_chunks_W{workload_id}{file_suffix}.pdf"
        filepath = output_dir / filename

        # Converti ChunkSize a stringa per la legenda
        group['ChunkSize_str'] = "Chunk=" + group['ChunkSize'].astype(str)

        fig = px.line(group,
                     x='NumThreads',
                     y='Speedup',
                     color='ChunkSize_str', # Usa stringa per rendere più leggibile la legenda
                     markers=True,
                     title=title,
                     labels={'NumThreads': 'Number of Threads',
                            'Speedup': 'Speedup (relative to Sequential)',
                            'ChunkSize_str': 'Chunk Size'})

        # Assicura che l'asse X mostri correttamente i thread
        unique_threads = sorted(group['NumThreads'].unique())
        if len(unique_threads) < 8:
            fig.update_xaxes(type='category')
        else:
            fig.update_xaxes(type='linear', dtick=2 if max(unique_threads) <= 16 else 4)

        # Aggiungi linea orizzontale per speedup = 1 (baseline)
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Baseline", annotation_position="bottom right")

        # Stima la frazione sequenziale analizzando i dati di speedup
        max_threads = max(unique_threads)
        if max_threads > 1:
            # Trova il miglior speedup per il massimo numero di thread tra tutti i chunk size
            max_thread_data = group[group['NumThreads'] == max_threads]
            if not max_thread_data.empty:
                best_speedup = max_thread_data['Speedup'].max()
                if best_speedup > 1:
                    s = (1 - best_speedup/max_threads) / (1 - 1/max_threads)
                    s = max(0.01, min(0.99, s))  # Limita a valori ragionevoli

                    # Aggiungi curva di Amdahl
                    amdahl_x = list(range(1, max(unique_threads) + 1)) if len(unique_threads) < 8 else np.linspace(1, max(unique_threads), 100)
                    amdahl_y = [1 / (s + (1-s)/n) for n in amdahl_x]

                    fig.add_trace(go.Scatter(
                        x=amdahl_x,
                        y=amdahl_y,
                        mode='lines',
                        line=dict(color='red', dash='dash', width=1.5),
                        name=f"Amdahl's Law (s={s:.2f})",
                        showlegend=True
                    ))

        fig.update_layout(width=width, height=height)
        try:
            fig.write_image(filepath, format="pdf")
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  ERROR saving {filepath}: {e}")


def plot_scheduler_heatmaps(df, plot_dir, show_speedup=True, file_suffix="", width=800, height=800):
    """Genera heatmaps per visualizzare le performance dei diversi scheduler in funzione
    del numero di thread e chunk size.
    """
    print(f"Plotting scheduler performance heatmaps ({'speedup' if show_speedup else 'execution time'})...")
    output_dir = plot_dir / "scheduler_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Considera solo scheduler che usano chunk size
    relevant_schedulers = ["Static Block", "Static Block-Cyclic", "Dynamic"]

    # Prepara i dati
    metric = 'Speedup' if show_speedup else 'ExecutionTimeMs'
    metric_label = 'Speedup' if show_speedup else 'Execution Time (ms)'

    # Filtra dati validi
    df_filtered = df.dropna(subset=['ChunkSize', 'NumThreads', metric])
    if not show_speedup:  # Se mostro execution time, filtra valori > 0
        df_filtered = df_filtered[df_filtered['ExecutionTimeMs'] > 0]

    df_filtered = df_filtered[df_filtered['ChunkSize'] > 0]  # Solo chunk size numerici > 0
    df_filtered = df_filtered[df_filtered['SchedulerName'].isin(relevant_schedulers)]

    # Converti a int
    df_filtered['ChunkSize'] = df_filtered['ChunkSize'].astype(int)
    df_filtered['NumThreads'] = df_filtered['NumThreads'].astype(int)

    # Per ogni workload e scheduler, genera un heatmap
    for workload_id, workload_group in df_filtered.groupby('WorkloadID'):
        workload_desc = workload_group['WorkloadDescription'].iloc[0]

        for scheduler_name, scheduler_group in workload_group.groupby('SchedulerName'):
            # Pivotta i dati per ottenere una matrice thread x chunk
            pivot_data = scheduler_group.pivot_table(
                index='ChunkSize',
                columns='NumThreads',
                values=metric,
                aggfunc='mean'  # In caso di duplicati, usa la media
            )

            # Ordina gli indici per avere un display più comprensibile
            pivot_data = pivot_data.sort_index()

            # Crea il plot
            title = f"{scheduler_name} {metric_label} Heatmap - {workload_desc}"
            filename = f"heatmap_{metric.lower()}_{scheduler_name.lower().replace(' ', '_')}_W{workload_id}{file_suffix}.pdf"
            filepath = output_dir / filename

            # Scegli la colorscale appropriata per il tipo di metrica
            # Per speedup, più alto è meglio (viridis)
            # Per execution time, più basso è meglio (viridis_r - invertito)
            colorscale = 'Viridis' if show_speedup else 'Viridis_r'

            fig = px.imshow(
                pivot_data,
                labels=dict(
                    x="Number of Threads",
                    y="Chunk Size",
                    color=metric_label
                ),
                x=pivot_data.columns.tolist(),
                y=pivot_data.index.tolist(),
                color_continuous_scale=colorscale,
                title=title,
                aspect="auto"
            )

            # # Aggiungi annotazioni con i valori per ogni cella
            # for i, row in enumerate(pivot_data.index):
            #     for j, col in enumerate(pivot_data.columns):
            #         value = pivot_data.iloc[i, j]
            #         # Formatta con 2 decimali
            #         text = f"{value:.2f}" if not pd.isna(value) else ""
            #         fig.add_annotation(
            #             x=col, y=row, text=text,
            #             showarrow=False,
            #             font=dict(color="white" if abs(value) > (pivot_data.max().max() / 2) else "black")
            #         )

            fig.update_layout(width=width, height=height)
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
    parser.add_argument('--dynamic-chunks-comparison', action='store_true',
                        help="Plot Dynamic scheduler performance with multiple chunk sizes")
    parser.add_argument('--blockcyclic-chunks-comparison', action='store_true',
                        help="Plot Static Block-Cyclic scheduler performance with multiple chunk sizes")
    parser.add_argument('--scheduler-heatmaps', action='store_true',
                        help="Plot heatmaps of scheduler performance (speedup) across thread counts and chunk sizes")
    # parser.add_argument('--scheduler-heatmaps-time', action='store_true',
    #                     help="Plot heatmaps of execution time across thread counts and chunk sizes")
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

    # print("Data loaded and preprocessed:")
    # print(df.info())
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
        'dynamic_chunks_comparison': generate_all or args.dynamic_chunks_comparison,
        'blockcyclic_chunks_comparison': generate_all or args.blockcyclic_chunks_comparison,
        'scheduler_heatmaps': generate_all or args.scheduler_heatmaps,
        # 'scheduler_heatmaps_time': generate_all or args.scheduler_heatmaps_time,
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

    # if plots_to_generate['time_vs_threads_log']:
    #     plot_time_vs_threads(df.copy(), plot_dir, args.fixed_chunk, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_speedup']:
        plot_chunk_impact_speedup(df.copy(), plot_dir, width=plot_width, height=plot_height)

    if plots_to_generate['chunk_impact_time']:
        plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=False, width=plot_width, height=plot_height)

    # if plots_to_generate['chunk_impact_time_log']:
    #     plot_chunk_impact_time(df.copy(), plot_dir, use_log_scale=True, file_suffix="_log", width=plot_width, height=plot_height)

    if plots_to_generate['dynamic_chunks_comparison']:
        plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Dynamic", width=plot_width, height=plot_height)

    if plots_to_generate['blockcyclic_chunks_comparison']:
        plot_scheduler_chunk_comparison(df.copy(), plot_dir, "Static Block-Cyclic", width=plot_width, height=plot_height)

    if plots_to_generate['scheduler_heatmaps']:
        plot_scheduler_heatmaps(df.copy(), plot_dir, show_speedup=True, width=plot_width, height=plot_height)

    if plots_to_generate['scheduler_heatmaps_time']:
        plot_scheduler_heatmaps(df.copy(), plot_dir, show_speedup=False, width=plot_width, height=plot_height)

    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()
