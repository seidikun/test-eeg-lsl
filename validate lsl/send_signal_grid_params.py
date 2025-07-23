import pandas as pd
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

# Parâmetros gerais
frequencias     = [40, 80, 320, 1280]  # Hz
num_canais_list = [5]
duracao         = 20  # segundos

freq_markers    = {'A': 0.7, 'B': 1.2, 'C': 2.0}
num_canais_max  = max(num_canais_list)
freq_max        = max(frequencias)

info        = StreamInfo('TestStream', 'EEG', num_canais_max, freq_max, 'float32', 'stream_global')
outlet      = StreamOutlet(info)
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'marker_global')
marker_outlet = StreamOutlet(marker_info)

sequencia_testes = []
blocos_param = []

# --- CRIAÇÃO DO ARQUIVO DE PARÂMETROS DOS BLOCOS ---
ordem = 1
for freq in frequencias:
    for num_canais in num_canais_list:
        bloc_id = f'{freq}Hz_{num_canais}ch'
        blocos_param.append({'ordem': ordem, 'tipo': 'START', 'freq': freq, 'num_canais': num_canais, 'marcador': f'START_{bloc_id}'})
        blocos_param.append({'ordem': ordem, 'tipo': 'END',   'freq': freq, 'num_canais': num_canais, 'marcador': f'END_{bloc_id}'})
        ordem += 1

df_blocos = pd.DataFrame(blocos_param)
df_blocos.to_csv('parametros_blocos.csv', index=False)
print('Arquivo parametros_blocos.csv salvo com os blocos esperados.\n')

# --- Transmissão ---
n_transmissoes = len(frequencias) * len(num_canais_list)
tempo_total = n_transmissoes * duracao
minutos, segundos = tempo_total // 60, tempo_total % 60
print(f"\nEstimativa de tempo total de execução: {tempo_total:.0f} segundos ({int(minutos)} min {int(segundos)} s)\n")

input("Pressione ENTER para iniciar todas as transmissões...\n")

for freq in frequencias:
    for num_canais in num_canais_list:
        t = np.arange(0, duracao, 1/freq)
        sinais = []
        for ch in range(num_canais):
            sinais.append(np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + 0.2*np.random.randn(len(t)))
        sinais = np.stack(sinais, axis=1)

        tempos_marcadores = []
        for label, intervalo in freq_markers.items():
            tempos = np.arange(0, duracao, intervalo)
            tempos_marcadores.extend([{'tempo': tempo, 'identidade': label} for tempo in tempos])
        tempos_marcadores.sort(key=lambda x: x['tempo'])
        prox_marker = 0

        print(f"\n>>> Iniciando Transmissão | {freq}Hz | {num_canais} canais <<<")
        sequencia_testes.append({'frequencia': freq, 'num_canais': num_canais, 'duracao': duracao})

        start_time = time.time()
        for i in range(len(t)):
            sample  = sinais[i, :].tolist()
            sample += [0.0] * (num_canais_max - num_canais)
            lsl_timestamp = start_time + t[i]

            if i == 0:
                marker_outlet.push_sample([f'START_{freq}Hz_{num_canais}ch'], lsl_timestamp)

            outlet.push_sample(sample, lsl_timestamp)

            while (prox_marker < len(tempos_marcadores) and abs(t[i] - tempos_marcadores[prox_marker]['tempo']) < (1/freq)/2):
                marker_outlet.push_sample([tempos_marcadores[prox_marker]['identidade']], lsl_timestamp)
                prox_marker += 1

            if i < len(t) - 1:
                dt = t[i+1] - t[i]
                if dt > 0:
                    time.sleep(dt)

        # ENVIA END após o término do loop, sempre!
        lsl_timestamp_end = time.time()
        marker_outlet.push_sample([f'END_{freq}Hz_{num_canais}ch'], lsl_timestamp_end)
        print(f"END enviado para {freq}Hz_{num_canais}ch @ {lsl_timestamp_end:.3f}")
        time.sleep(0.1)  # Garante flush do marker

        df_sinais                  = pd.DataFrame(sinais, columns=[f'ch{i+1}' for i in range(num_canais)])
        df_sinais['lsl_timestamp'] = start_time + t
        df_sinais['frequencia']    = freq
        df_sinais['num_canais']    = num_canais

        nome_arquivo = f'sinais_enviados_{freq}Hz_{num_canais}ch.csv'
        df_sinais.to_csv(nome_arquivo, index=False)
        print(f"Sinais salvos em {nome_arquivo}")

print("\nTodas as transmissões concluídas.")
time.sleep(1)  # Garante flush do LSL ao final do script
df_sequencia = pd.DataFrame(sequencia_testes)
df_sequencia.to_csv('sequencia_testes.csv', index=False)
print("Sequência de testes salva em 'sequencia_testes.csv'.")
