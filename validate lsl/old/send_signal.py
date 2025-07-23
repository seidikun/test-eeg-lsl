from pylsl import StreamInfo, StreamOutlet
import numpy as np
import pandas as pd
import time

FS       = 2500
N_CHANS  = 10
SIG_TYPE = 'complexo'
DURATION = 10

t = np.arange(0, DURATION, 1/FS)
sinais = []
for ch in range(N_CHANS):
    if SIG_TYPE == 'seno':
        sinais.append(np.sin(2 * np.pi * (5 + ch*5) * t))
    else:
        sinais.append(np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + 0.2*np.random.randn(len(t)))
sinais = np.stack(sinais, axis=1)

info = StreamInfo('TestStream', 'EEG', N_CHANS, FS, 'float32', 'myuid34234')
outlet = StreamOutlet(info)
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'markeruid')
marker_outlet = StreamOutlet(marker_info)

# Frequências dos marcadores (em segundos)
freq_A = 0.7
freq_B = 1.2
freq_C = 2.0

# Gera todos os tempos possíveis para cada marcador
tempos_A = np.arange(0, DURATION, freq_A)
tempos_B = np.arange(0, DURATION, freq_B)
tempos_C = np.arange(0, DURATION, freq_C)

# Cria lista de marcadores
marcadores = (
    [{'tempo': tempo, 'identidade': 'A'} for tempo in tempos_A] +
    [{'tempo': tempo, 'identidade': 'B'} for tempo in tempos_B] +
    [{'tempo': tempo, 'identidade': 'C'} for tempo in tempos_C]
)
# Ordena todos por tempo
marcadores.sort(key=lambda x: x['tempo'])
prox_marker = 0

input("\nPronto para transmitir.\nPressione ENTER para iniciar o envio dos sinais...\n")

print(f"Enviando '{SIG_TYPE}' com {N_CHANS} canais a {FS}Hz por {DURATION}s")
samples_sent = []
markers_sent = []

start_time = time.time()

try:
    for i in range(len(t)):
        sample = sinais[i, :].tolist()
        lsl_timestamp = start_time + t[i]
        outlet.push_sample(sample, lsl_timestamp)
        samples_sent.append(sample + [lsl_timestamp])

        while prox_marker < len(marcadores) and abs(t[i] - marcadores[prox_marker]['tempo']) < (1/FS)/2:
            marker = marcadores[prox_marker]['identidade']
            marker_outlet.push_sample([marker], lsl_timestamp)
            print(f"Enviou marcador: {marker} em t={t[i]:.2f}s")
            markers_sent.append({'sample_idx': i, 'timestamp': lsl_timestamp, 'marker': marker})
            prox_marker += 1

        if i % FS == 0:
            print(f"t={t[i]:.2f}s | sample={sample}")

        if i < len(t) - 1:
            dt = t[i+1] - t[i]
            if dt > 0:
                time.sleep(dt)
finally:
    print("Transmissão finalizada.")
    df_env = pd.DataFrame(samples_sent, columns=[f'ch{i+1}' for i in range(N_CHANS)] + ['lsl_timestamp'])
    df_env.to_csv('sinais_enviados.csv', index=False)
    pd.DataFrame(markers_sent).to_csv('marcadores_enviados.csv', index=False)
    print("Sinal e marcadores enviados foram salvos em arquivos.")
    del outlet
    del marker_outlet
    print("Outlets deletados. Streams devem desaparecer.")
