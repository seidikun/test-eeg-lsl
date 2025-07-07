import pandas as pd
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

# Parâmetros gerais
frequencias = [20, 50, 100, 200, 1000]
num_canais_list = [1, 5, 15, 20]
duracao = 15  # segundos
repeticoes = 5
sig_type = 'complexo'

freq_markers = {'A': 0.7, 'B': 1.2, 'C': 2.0}

num_canais_max = max(num_canais_list)
freq_max = max(frequencias)

info = StreamInfo('TestStream', 'EEG', num_canais_max, freq_max, 'float32', 'stream_global')
outlet = StreamOutlet(info)

marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'marker_global')
marker_outlet = StreamOutlet(marker_info)

sequencia_testes = []

input("\nPressione ENTER para iniciar todas as transmissões...\n")

for freq in frequencias:
 for num_canais in num_canais_list:
  for rep in range(1, repeticoes + 1):

   t = np.arange(0, duracao, 1/freq)
   sinais = []

   for ch in range(num_canais):
    if sig_type == 'seno':
     sinais.append(np.sin(2 * np.pi * (5 + ch*5) * t))
    else:
     sinais.append(np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + 0.2*np.random.randn(len(t)))

   sinais = np.stack(sinais, axis=1)

   tempos_marcadores = []
   for label, intervalo in freq_markers.items():
    tempos = np.arange(0, duracao, intervalo)
    tempos_marcadores.extend([{'tempo': tempo, 'identidade': label} for tempo in tempos])
   tempos_marcadores.sort(key=lambda x: x['tempo'])
   prox_marker = 0

   print(f"\n>>> Iniciando Transmissão {rep}/5 | {freq}Hz | {num_canais} canais <<<")
   sequencia_testes.append({'frequencia': freq, 'num_canais': num_canais, 'repeticao': rep, 'duracao': duracao})

   start_time = time.time()
   registros = []

   for i in range(len(t)):
    sample = sinais[i, :].tolist()
    sample += [0.0] * (num_canais_max - num_canais)
    lsl_timestamp = start_time + t[i]

    marker_to_send = ''

    if i == 0:
     marker_to_send = f'START_{freq}Hz_{num_canais}ch_rep{rep}'
     marker_outlet.push_sample([marker_to_send], lsl_timestamp)

    while prox_marker < len(tempos_marcadores) and abs(t[i] - tempos_marcadores[prox_marker]['tempo']) < (1/freq)/2:
     marker_to_send = tempos_marcadores[prox_marker]['identidade']
     marker_outlet.push_sample([marker_to_send], lsl_timestamp)
     prox_marker += 1

    outlet.push_sample(sample, lsl_timestamp)

    registros.append({
     **{f'ch{k+1}': sample[k] for k in range(num_canais)},
     'lsl_timestamp': lsl_timestamp,
     'frequencia': freq,
     'num_canais': num_canais,
     'repeticao': rep,
     'marker': marker_to_send
    })

    if i < len(t) - 1:
     dt = t[i+1] - t[i]
     if dt > 0:
      time.sleep(dt)

   end_marker = f'END_{freq}Hz_{num_canais}ch_rep{rep}'
   marker_outlet.push_sample([end_marker], start_time + duracao)

   df_sinais = pd.DataFrame(registros)
   nome_arquivo = f'sinais_enviados_{freq}Hz_{num_canais}ch_rep{rep}.csv'
   df_sinais.to_csv(nome_arquivo, index=False)
   print(f"Sinais salvos em {nome_arquivo}")

   time.sleep(2)

print("\nTodas as transmissões concluídas.")
pd.DataFrame(sequencia_testes).to_csv('sequencia_testes.csv', index=False)
print("Sequência de testes salva.")
