import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Carregar dados ---
recv = pd.read_csv('dados_recebidos_teste_params.csv')
recv = recv[recv['lsl_timestamp'] > 1e8].reset_index(drop=True)
recv_signal = recv[recv['marker'].isna() | (recv['marker'] == '')].reset_index(drop=True)
recv_markers = recv[recv['marker'].notna() & (recv['marker'] != '')][['lsl_timestamp', 'marker']].reset_index(drop=True)
recv_markers['marker'] = recv_markers['marker'].astype(str)

sequencia = pd.read_csv('sequencia_testes.csv')
send_files = {
    f"{row['frequencia']}Hz_{row['num_canais']}ch": f
    for f in glob.glob('sinais_enviados_*.csv')
    for _, row in sequencia.iterrows()
    if f.endswith(f"{row['frequencia']}Hz_{row['num_canais']}ch.csv")
}

# --- Encontrar blocos (início/fim) ---
starts = recv_markers[recv_markers['marker'].str.startswith('START')]
ends   = recv_markers[recv_markers['marker'].str.startswith('END')]
blocks = []
for _, start in starts.iterrows():
    after = ends[ends['lsl_timestamp'] > start['lsl_timestamp']]
    if not after.empty:
        blocks.append({
            'start': start['lsl_timestamp'],
            'end': after['lsl_timestamp'].iloc[0],
            'label': start['marker']
        })

print(f"Total de blocos detectados: {len(blocks)}")

# --- Processamento e plot ---
plt.style.use('default')
fig, axs = plt.subplots(len(blocks), 4, figsize=(14, 3 * len(blocks)), squeeze=False)

for i, blk in enumerate(blocks):
    try:
        freq = int(blk['label'].split('START_')[1].split('Hz')[0])
        chans = int(blk['label'].split('Hz_')[1].split('ch')[0])
        key = f"{freq}Hz_{chans}ch"
        send_file = send_files.get(key)
    except Exception as e:
        print(f"Erro ao extrair/corresponder: {e}")
        continue

    if not send_file:
        continue

    sdata = pd.read_csv(send_file)
    sdata = sdata[(sdata['lsl_timestamp'] >= blk['start']) & (sdata['lsl_timestamp'] <= blk['end'])].reset_index(drop=True)
    rdata = recv_signal[(recv_signal['lsl_timestamp'] >= blk['start']) & (recv_signal['lsl_timestamp'] <= blk['end'])].reset_index(drop=True)
    if sdata.empty or rdata.empty: continue

    n = min(len(sdata), len(rdata))
    if n < 2: continue

    periodo = 1 / freq
    periodo_ms = 1000 / freq

    # --- Tempos recebidos (corrigido e local) ---
    if 'lsl_timestamp_corr' in rdata.columns:
        tempo_recebido_corr = rdata['lsl_timestamp_corr'].values[:n]
    else:
        tempo_recebido_corr = None
    tempo_recebido = rdata['local_time'].values[:n]

    # --- Loop time e delay acumulado só cresce ---
    loop_times = rdata['loop_time_ms'].values[:n] if 'loop_time_ms' in rdata else np.zeros(n)
    delay_acumulado = np.zeros(n)
    for j in range(1, n):
        excesso = loop_times[j-1] - periodo_ms
        if excesso > 0:
            delay_acumulado[j] = delay_acumulado[j-1] + excesso
        else:
            delay_acumulado[j] = delay_acumulado[j-1]
    delay_acum_media = np.mean(delay_acumulado)

    # --- Subplots padrão para diagnóstico ---
    send_diffs = np.diff(sdata['lsl_timestamp'].values[:n])
    recv_diffs = np.diff(tempo_recebido)

    ax1, ax2, ax3, ax4 = axs[i]

    # Δt Send e Recv
    ax1.plot(send_diffs, alpha=0.8, color='blue', label='Send Δt')
    ax1.plot(recv_diffs, alpha=0.8, color='orange', label='Recv Δt')
    ax1.axhline(send_diffs.mean(), ls='--', color='blue', label=f'Média Send: {send_diffs.mean():.5f}s')
    ax1.axhline(recv_diffs.mean(), ls='--', color='orange', label=f'Média Recv: {recv_diffs.mean():.5f}s')
    ax1.set_title(f'Delta t | {freq}Hz {chans}ch')
    ax1.legend()

    # Tempos absolutos
    st = sdata['lsl_timestamp'].values[:len(send_diffs)+1]
    rt = tempo_recebido[:len(recv_diffs)+1]
    ax2.plot(st - st[0], label='Send', color='blue', alpha=0.7)
    ax2.plot(rt - rt[0], label='Recv (local_time)', color='orange', alpha=0.7)
    if tempo_recebido_corr is not None:
        rtc = tempo_recebido_corr[:len(recv_diffs)+1]
        ax2.plot(rtc - rtc[0], label='Recv (clock corr)', color='purple', alpha=0.7)
    ax2.set_title('Tempos absolutos')
    ax2.legend()

    # Loop time
    ax3.plot(loop_times, color='green', alpha=0.6)
    ax3.axhline(np.mean(loop_times), ls='--', color='green', label=f'Média: {np.mean(loop_times):.1f} ms')
    ax3.axhline(periodo_ms, ls=':', color='red', label=f'Máximo: {periodo_ms:.1f} ms')
    ax3.set_title('Loop time (ms)')
    ax3.legend()

    # Delay acumulado só cresce
    ax4.plot(delay_acumulado, color='purple', alpha=0.8, label='Delay acumulado')
    ax4.axhline(delay_acum_media, color='purple', ls='--', label=f'Média: {delay_acum_media:.1f} ms')
    ax4.set_title('Delay acumulado (ms)')
    ax4.set_ylabel('ms')
    ax4.legend()

plt.suptitle('Análise por Frequência e N Canais', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('analise_grid_resultados.pdf')
plt.show()
