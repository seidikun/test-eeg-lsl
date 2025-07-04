import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob

recv = pd.read_csv('dados_recebidos_teste_params.csv', on_bad_lines='skip', low_memory=False)
recv = recv[recv['lsl_timestamp'] > 1e8].reset_index(drop=True)
recv_chans = [col for col in recv.columns if col.startswith('ch')]

recv_markers = recv[~recv['marker'].isna() & (recv['marker'] != '')][['lsl_timestamp', 'marker']].reset_index(drop=True)
recv_signal = recv[recv['marker'].isna() | (recv['marker'] == '')].reset_index(drop=True)

sequencia = pd.read_csv('sequencia_testes.csv')
files_send = {f"{row['frequencia']}Hz_{row['num_canais']}ch_rep{row['repeticao']}": f
              for f in glob.glob('sinais_enviados_*.csv')
              for _, row in sequencia.iterrows()
              if f.endswith(f"{row['frequencia']}Hz_{row['num_canais']}ch_rep{row['repeticao']}.csv")}

start_markers = recv_markers[recv_markers['marker'].str.startswith('START')]
end_markers = recv_markers[recv_markers['marker'].str.startswith('END')]

blocks = []
for idx, start in start_markers.iterrows():
    end_times = end_markers[end_markers['lsl_timestamp'] > start['lsl_timestamp']]
    if not end_times.empty:
        end_time = end_times['lsl_timestamp'].iloc[0]
        label = start['marker']
        blocks.append({'start_time': start['lsl_timestamp'], 'end_time': end_time, 'label': label})

print(f"Total de blocos detectados: {len(blocks)}")

param_grid = {}

for block in blocks:
    start_t, end_t, label = block['start_time'], block['end_time'], block['label']

    try:
        freq = int(label.split('START_')[1].split('Hz')[0])
        canais = int(label.split('Hz_')[1].split('ch')[0])
        rep = int(label.split('rep')[1])
    except (IndexError, ValueError):
        continue

    if rep != 1:
        continue

    chave = f"{freq}Hz_{canais}ch_rep{rep}"
    send_file = files_send.get(chave)

    if not send_file:
        continue

    send_block = pd.read_csv(send_file)
    send_block = send_block[(send_block['lsl_timestamp'] >= start_t) & (send_block['lsl_timestamp'] <= end_t)].reset_index(drop=True)
    recv_block = recv_signal[(recv_signal['lsl_timestamp'] >= start_t) & (recv_signal['lsl_timestamp'] <= end_t)].reset_index(drop=True)

    if send_block.empty or recv_block.empty:
        continue

    min_len = min(len(send_block), len(recv_block))
    if min_len < 2:
        continue

    send_diffs = np.diff(send_block['lsl_timestamp'].values[:min_len])

    if 'local_time' in recv_block.columns:
        recv_times = recv_block['local_time'].values[:min_len]
    else:
        recv_times = recv_block['lsl_timestamp'].values[:min_len]

    recv_diffs = np.diff(recv_times)

    param_grid[(freq, canais)] = (send_diffs, recv_diffs, send_block, recv_block)

param_list = sorted(param_grid.keys())

with plt.rc_context({'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8}):
    fig, axs = plt.subplots(len(param_list), 3, figsize=(10, 3 * len(param_list)), squeeze=False)

    for idx, (freq, canais) in enumerate(param_list):
        send_diffs, recv_diffs, send_block, recv_block = param_grid[(freq, canais)]

        shared_min = min(send_diffs.min(), recv_diffs.min())
        shared_max = max(send_diffs.max(), recv_diffs.max())

        ax_send = axs[idx][0]
        ax_recv = axs[idx][1]
        ax_time = axs[idx][2]

        ax_send.plot(send_diffs, color='blue', alpha=0.7)
        ax_send.axhline(np.mean(send_diffs), linestyle='--', color='blue', label=f'Média: {np.mean(send_diffs):.5f}s')
        ax_send.set_ylim(shared_min, shared_max)
        ax_send.set_title(f'Send Δt | {freq}Hz {canais}ch')
        ax_send.legend()

        ax_recv.plot(recv_diffs, color='orange', alpha=0.7)
        ax_recv.axhline(np.mean(recv_diffs), linestyle='--', color='orange', label=f'Média: {np.mean(recv_diffs):.5f}s')
        ax_recv.set_ylim(shared_min, shared_max)
        ax_recv.set_title(f'Recv Δt (local_time) | {freq}Hz {canais}ch')
        ax_recv.legend()

        send_times = send_block['lsl_timestamp'].values[:len(send_diffs)+1]
        recv_times = recv_block['local_time'].values[:len(recv_diffs)+1]

        ax_time.plot(send_times - send_times[0], label='Send Times', color='blue', alpha=0.7)
        ax_time.plot(recv_times - recv_times[0], label='Recv Times', color='orange', alpha=0.7)
        ax_time.set_title(f'Tempos Absolutos')
        ax_time.legend()

    plt.suptitle('Análise por Frequência e Número de Canais (Rep 1)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('analise_grid_resultados.pdf')
    plt.show()
