import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import defaultdict

recv = pd.read_csv('dados_recebidos_teste_params.csv', on_bad_lines='skip', low_memory=False)
recv = recv[recv['lsl_timestamp'] > 1e8].reset_index(drop=True)
recv_markers = recv[~recv['marker'].isna() & (recv['marker'] != '')][['lsl_timestamp', 'marker', 'local_time']].reset_index(drop=True)
recv_signal = recv[recv['marker'].isna() | (recv['marker'] == '')].reset_index(drop=True)

sequencia = pd.read_csv('sequencia_testes.csv')
files_send = {f"{row['frequencia']}Hz_{row['num_canais']}ch_rep{row['repeticao']}": f
              for f in glob.glob('sinais_enviados_*.csv')
              for _, row in sequencia.iterrows()
              if f.endswith(f"{row['frequencia']}Hz_{row['num_canais']}ch_rep{row['repeticao']}.csv")}

start_markers = recv_markers[recv_markers['marker'].str.startswith('START')].reset_index(drop=True)
end_markers = recv_markers[recv_markers['marker'].str.startswith('END')].reset_index(drop=True)

blocks = []
for _, start_row in start_markers.iterrows():
    start_time = start_row['lsl_timestamp']
    end_times = end_markers[end_markers['lsl_timestamp'] > start_time]
    if not end_times.empty:
        end_time = end_times.iloc[0]['lsl_timestamp']
        label = start_row['marker']
        blocks.append({'label': label, 'start_time': start_time, 'end_time': end_time})

print("Total de blocos identificados:", len(blocks))

param_delays = defaultdict(lambda: defaultdict(list))

for block in blocks:
    label = block['label']
    try:
        freq = int(label.split('START_')[1].split('Hz')[0])
        canais = int(label.split('Hz_')[1].split('ch')[0])
        rep = int(label.split('rep')[1])
    except:
        continue

    block_data = recv[(recv['lsl_timestamp'] >= block['start_time']) & (recv['lsl_timestamp'] <= block['end_time'])]
    markers_block = block_data[~block_data['marker'].isna() & (block_data['marker'] != '')][['lsl_timestamp', 'marker', 'local_time']].reset_index(drop=True)

    for marker_name in ['A', 'B', 'C']:
        marker_rows = markers_block[markers_block['marker'] == marker_name]
        if marker_rows.empty:
            continue
        delays = marker_rows['local_time'].values - marker_rows['lsl_timestamp'].values
        param_delays[(freq, canais)][marker_name].extend(list(zip(delays, marker_rows['lsl_timestamp'].values)))

fig, axs = plt.subplots(len(param_delays), 3, figsize=(12, 2.5 * len(param_delays)), squeeze=False)

# Determinar o range global do eixo x
all_delays = [delay for markers in param_delays.values() for delays in markers.values() for delay, _ in delays]
xmin, xmax = min(all_delays) - 0.05, max(all_delays) + 0.05

for idx, ((freq, canais), marker_dict) in enumerate(sorted(param_delays.items())):
    for j, marker_name in enumerate(['A', 'B', 'C']):
        delay_time_pairs = marker_dict.get(marker_name, [])
        ax = axs[idx][j]
        if delay_time_pairs:
            delays, times = zip(*delay_time_pairs)
            norm_times = (np.array(times) - min(times)) / (max(times) - min(times) + 1e-9)
            colors = plt.cm.viridis(norm_times)
            for delay, color in zip(delays, colors):
                ax.vlines(delay, 0, 1, color=color, alpha=0.7)
            ax.axvline(0, color='black', linestyle='--')
            ax.set_xlim(xmin, xmax)
        ax.set_title(f'{freq}Hz {canais}ch | Marker {marker_name}')
        ax.set_yticks([])
        ax.set_xlabel('Delay (s)')

axs[0][0].set_ylabel('Delays (vlines)')
plt.suptitle('Delays calculados como local_time - lsl_timestamp', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('delays_vline_localtime_vs_lsl.pdf')
plt.show()

print("\nAnálise concluída.")
