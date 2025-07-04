from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import numpy as np
import time

def main():
 print("Buscando streams de EEG e Markers (pelo 'type')...")

 eeg_streams = resolve_byprop('type', 'EEG', timeout=10)
 if not eeg_streams:
  print("Nenhum stream EEG encontrado!")
  return
 eeg_inlet = StreamInlet(eeg_streams[-1])

 marker_streams = resolve_byprop('type', 'Markers', timeout=10)
 if not marker_streams:
  print("Nenhum stream Markers encontrado!")
  return
 marker_inlet = StreamInlet(marker_streams[-1])

 n_chans = eeg_inlet.channel_count

 csv_path = 'dados_recebidos_teste_params.csv'
 columns = ['lsl_timestamp', 'local_time'] + [f'ch{i+1}' for i in range(n_chans)] + ['marker']
 pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

 print(f"\nGravação iniciada: {csv_path}")
 print("Recebendo dados... Pressione Ctrl+C para parar.\n")

 primeira_amostra = True

 try:
  while True:
   now = time.time()
   sample, ts = eeg_inlet.pull_sample(timeout=0.01)

   if sample is not None:
    row = {'lsl_timestamp': ts, 'local_time': now}
    for i, val in enumerate(sample):
     row[f'ch{i+1}'] = val
    row['marker'] = ''
    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

    if primeira_amostra:
     print(f"EEG conectado: {n_chans} canais | Exemplo de amostra: {sample}")
     primeira_amostra = False

   while True:
    marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
    if marker is not None:
     now_marker = time.time()
     row = {'lsl_timestamp': marker_ts, 'local_time': now_marker}
     for i in range(n_chans):
      row[f'ch{i+1}'] = np.nan
     row['marker'] = marker[0]
     pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

     print(f"MARCADOR: {marker[0]} | t_LSL={marker_ts:.4f}")
    else:
     break

 except KeyboardInterrupt:
  print("\nFinalizando recepção.")
  print(f"Dados salvos em {csv_path}")

if __name__ == "__main__":
 main()
