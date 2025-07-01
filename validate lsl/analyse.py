import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate

# --- Carregar e filtrar arquivos ---
env = pd.read_csv('sinais_enviados.csv')
recv = pd.read_csv('dados_unificados.csv', on_bad_lines='skip')
recv = recv[recv['lsl_timestamp'] > 1e8].reset_index(drop=True)

env_chans = [col for col in env.columns if col.startswith('ch')]
recv_chans = [col for col in recv.columns if col.startswith('ch')]
n_chans = len(env_chans)
offset = 4

t0_env = env['lsl_timestamp'].iloc[0]
t0_recv = recv['lsl_timestamp'].iloc[0]
t_env = env['lsl_timestamp'] - t0_env

mask_signal = recv['marker'].isna() | (recv['marker'] == '')
recv_signal = recv[mask_signal].sort_values('lsl_timestamp').reset_index(drop=True)
t_recv = recv_signal['lsl_timestamp'] - t0_recv

env_markers = pd.read_csv('marcadores_enviados.csv')
recv_markers = recv[~(recv['marker'].isna()) & (recv['marker'] != '')][['lsl_timestamp', 'marker']].reset_index(drop=True)
marker_colors = dict(A='tab:red', B='tab:blue', C='tab:green')

# --- PLOT ENVIADO ---
plt.figure(figsize=(15, 2.5*n_chans))
for i, ch in enumerate(env_chans):
    plt.plot(t_env, env[ch] + i*offset, color='k')
for _, row in env_markers.iterrows():
    color = marker_colors.get(row['marker'], 'gray')
    x = row['timestamp'] - t0_env
    plt.axvline(x=x, color=color, linestyle='--', alpha=0.7)
plt.title("Sinais ENVIADOS (offset) + marcadores")
plt.xlabel("Tempo (s)")
plt.yticks([i*offset for i in range(n_chans)], [f'Ch {i+1}' for i in range(n_chans)])
plt.tight_layout()
plt.show()

# --- PLOT RECEBIDO: eixo LSL timestamp ---
plt.figure(figsize=(15, 2.5*n_chans))
for i, ch in enumerate(recv_chans):
    plt.plot(t_recv, recv_signal[ch] + i*offset, color='tab:orange')
for _, row in recv_markers.iterrows():
    color = marker_colors.get(row['marker'], 'gray')
    x = row['lsl_timestamp'] - t0_recv
    plt.axvline(x=x, color=color, linestyle='--', alpha=0.7)
plt.title("Sinais RECEBIDOS (offset, LSL timestamp) + marcadores")
plt.xlabel("Tempo (s) LSL timestamp")
plt.yticks([i*offset for i in range(n_chans)], [f'Ch {i+1}' for i in range(n_chans)])
plt.tight_layout()
plt.show()

# --- PLOT RECEBIDO: eixo local_time ---
if 'local_time' in recv_signal.columns:
    t0_local = recv_signal['local_time'].iloc[0]
    t_local = recv_signal['local_time'] - t0_local
    plt.figure(figsize=(15, 2.5*n_chans))
    for i, ch in enumerate(recv_chans):
        plt.plot(t_local, recv_signal[ch] + i*offset, color='tab:purple')
    for _, row in recv_markers.iterrows():
        color = marker_colors.get(row['marker'], 'gray')
        idx = np.argmin(np.abs(recv_signal['lsl_timestamp'] - row['lsl_timestamp']))
        x = recv_signal['local_time'].iloc[idx] - t0_local
        plt.axvline(x=x, color=color, linestyle='--', alpha=0.7)
    plt.title("Sinais RECEBIDOS (offset, local_time) + marcadores")
    plt.xlabel("Tempo (s) local_time")
    plt.yticks([i*offset for i in range(n_chans)], [f'Ch {i+1}' for i in range(n_chans)])
    plt.tight_layout()
    plt.show()

# --- Correlação e Lag ótimo (LSL timestamp) ---
print("\nCorrelação Pearson e lag ótimo (timestamp LSL):")
fs = 250
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
    r, _ = pearsonr(x, y)
    x0, y0 = x - np.mean(x), y - np.mean(y)
    corr = correlate(y0, x0, mode='full')
    lags = np.arange(-len(x0)+1, len(y0))
    best_lag = lags[np.argmax(corr)]
    lag_ms = best_lag * 1000 / fs
    print(f"{ch}: r={r:.4f}, lag={best_lag} amostras ({lag_ms:.1f} ms)")

# --- Correlação e Lag ótimo (local_time) ---
if 'local_time' in recv_signal.columns:
    print("\nCorrelação Pearson e lag ótimo (clock local):")
    t0_local = recv_signal['local_time'].iloc[0]
    t_local = recv_signal['local_time'] - t0_local
    t_env_uniform = np.arange(n) / fs
    for ch in env_chans:
        y_full = recv_signal[ch].values[:len(t_local)]
        t_full = t_local.values[:len(t_local)]
        mask = ~np.isnan(y_full)
        t_good = t_full[mask]
        y_good = y_full[mask]
        if len(y_good) < 2:
            print(f"{ch}: Sinal recebido insuficiente (local_time).")
            continue
        y_interp = np.interp(t_env_uniform[:len(y_good)], t_good, y_good)
        x = env[ch].values[:len(y_interp)]
        y = y_interp
        mask2 = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask2]
        y = y[mask2]
        if len(x) == 0 or len(y) == 0:
            print(f"{ch}: Sem dados válidos (local_time).")
            continue
        r, _ = pearsonr(x, y)
        x0, y0 = x - np.mean(x), y - np.mean(y)
        corr = correlate(y0, x0, mode='full')
        lags = np.arange(-len(x0)+1, len(y0))
        best_lag = lags[np.argmax(corr)]
        lag_ms = best_lag * 1000 / fs
        print(f"{ch}: r={r:.4f}, lag={best_lag} amostras ({lag_ms:.1f} ms) [local_time]")
