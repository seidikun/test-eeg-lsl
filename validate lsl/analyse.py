import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate

# --- Carregar os arquivos ---
env = pd.read_csv('sinais_enviados.csv')

# Em pandas >= 1.3.0
recv = pd.read_csv('dados_unificados.csv', on_bad_lines='skip')
recv.to_csv('dados_unificados_corrigido.csv', index=False)
print("Arquivo corrigido salvo como dados_unificados_corrigido.csv")


# Remove dados com timestamp inválido/muito pequeno
recv = recv[recv['lsl_timestamp'] > 1e8].reset_index(drop=True)

# Nome dos canais
env_chans = [col for col in env.columns if col.startswith('ch')]
recv_chans = [col for col in recv.columns if col.startswith('ch')]
n_chans = len(env_chans)
offset = 4

# Tempo relativo
t0_env = env['lsl_timestamp'].iloc[0]
t0_recv = recv['lsl_timestamp'].iloc[0]
t_env = env['lsl_timestamp'] - t0_env

# Apenas linhas de sinal recebido
mask_signal = recv['marker'].isna() | (recv['marker'] == '')
recv_signal = recv[mask_signal].sort_values('lsl_timestamp').reset_index(drop=True)
t_recv = recv_signal['lsl_timestamp'] - t0_recv

# Marcadores enviados e recebidos
env_markers = pd.read_csv('marcadores_enviados.csv')
recv_markers = recv[~(recv['marker'].isna()) & (recv['marker'] != '')][['lsl_timestamp', 'marker']].reset_index(drop=True)

marker_colors = dict(A='tab:red', B='tab:blue', C='tab:green')

# --- PLOT ENVIADO ---
plt.figure(figsize=(15, 2.5*n_chans))
yticks, yticklabels = [], []
for i, ch in enumerate(env_chans):
    plt.plot(t_env, env[ch] + i*offset, label=f'Enviado {ch}', color='k')
    yticks.append(i*offset)
    yticklabels.append(f'Ch {i+1}')
for _, row in env_markers.iterrows():
    color = marker_colors.get(row['marker'], 'gray')
    x = row['timestamp'] - t0_env
    plt.axvline(x=x, color=color, linestyle='--', alpha=0.7, lw=1)
    plt.text(x, yticks[-1]+offset/1.5, row['marker'], color=color, ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("Sinais ENVIADOS (empilhados com offset) e marcadores enviados")
plt.xlabel("Tempo (s)")
plt.yticks(yticks, yticklabels)
plt.tight_layout()
plt.show()

# --- PLOT RECEBIDO: eixo LSL timestamp ---
plt.figure(figsize=(15, 2.5*n_chans))
yticks, yticklabels = [], []
for i, ch in enumerate(recv_chans):
    plt.plot(t_recv, recv_signal[ch] + i*offset, label=f'Recebido {ch}', color='tab:orange')
    yticks.append(i*offset)
    yticklabels.append(f'Ch {i+1}')
for _, row in recv_markers.iterrows():
    color = marker_colors.get(row['marker'], 'gray')
    x = row['lsl_timestamp'] - t0_recv
    plt.axvline(x=x, color=color, linestyle='--', alpha=0.7, lw=1)
    plt.text(x, yticks[-1]+offset/1.5, row['marker'], color=color, ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("Sinais RECEBIDOS (empilhados com offset) e marcadores recebidos (LSL timestamp)")
plt.xlabel("Tempo (s) LSL timestamp")
plt.yticks(yticks, yticklabels)
plt.tight_layout()
plt.show()

# --- PLOT RECEBIDO: eixo local_time ---
if 'local_time' not in recv_signal.columns:
    print("\n[AVISO] Não há coluna 'local_time' no arquivo recebido!")
else:
    t0_local = recv_signal['local_time'].iloc[0]
    t_local = recv_signal['local_time'] - t0_local

    plt.figure(figsize=(15, 2.5*n_chans))
    yticks, yticklabels = [], []
    for i, ch in enumerate(recv_chans):
        plt.plot(t_local, recv_signal[ch] + i*offset, label=f'Recebido {ch}', color='tab:purple')
        yticks.append(i*offset)
        yticklabels.append(f'Ch {i+1}')
    for _, row in recv_markers.iterrows():
        color = marker_colors.get(row['marker'], 'gray')
        # Tenta buscar o local_time do marcador pelo índice mais próximo
        idx = np.argmin(np.abs(recv_signal['lsl_timestamp'] - row['lsl_timestamp']))
        x = recv_signal['local_time'].iloc[idx] - t0_local
        plt.axvline(x=x, color=color, linestyle='--', alpha=0.7, lw=1)
        plt.text(x, yticks[-1]+offset/1.5, row['marker'], color=color, ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.title("Sinais RECEBIDOS (empilhados com offset) e marcadores recebidos (local_time)")
    plt.xlabel("Tempo (s) local_time")
    plt.yticks(yticks, yticklabels)
    plt.tight_layout()
    plt.show()

# --- Correlação e Lag ótimo ---
print("\nCorrelação Pearson e lag ótimo (cross-correlation):")
fs = 250  # taxa de amostragem usada no experimento

# Garante mesmo número de amostras
n = min(len(env), len(recv_signal))
env_valid = env.iloc[:n].reset_index(drop=True)
recv_valid = recv_signal.iloc[:n].reset_index(drop=True)

for ch in env_chans:
    x = env_valid[ch].values
    y = recv_valid[ch].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0 or len(y) == 0:
        print(f"{ch}: Sem dados válidos.")
        continue

    # Correlação simples
    r, _ = pearsonr(x, y)

    # Lag ótimo (max cross-correlation)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    corr = correlate(y0, x0, mode='full')  # y em relação a x
    lags = np.arange(-len(x0) + 1, len(y0))
    best_lag = lags[np.argmax(corr)]
    lag_ms = best_lag * 1000 / fs

    print(f"{ch}: r = {r:.4f} | lag ótimo = {best_lag} amostras ({lag_ms:.1f} ms)")
