from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import numpy as np
import os
import time

def main():
    print("Buscando TestStream EEG (último da lista)...")
    eeg_streams = resolve_byprop('name', 'TestStream', timeout=10)
    if not eeg_streams:
        print("Nenhum stream EEG encontrado!")
        return
    eeg_inlet = StreamInlet(eeg_streams[-1])

    print("Buscando Markers (último da lista)...")
    marker_streams = resolve_byprop('name', 'Markers', timeout=10)
    if not marker_streams:
        print("Nenhum stream Markers encontrado!")
        return
    marker_inlet = StreamInlet(marker_streams[-1])

    # Cria arquivo e escreve cabeçalho
    n_chans = eeg_inlet.channel_count
    csv_path = 'dados_unificados.csv'
    columns = ['lsl_timestamp', 'local_time'] + [f'ch{i+1}' for i in range(n_chans)] + ['marker']
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    if write_header:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    print("Recebendo dados... pressione Ctrl+C para parar.")
    last_print_time = 0
    print_interval = 1/30  # mostra dados a cada 1/30s (30Hz)

    try:
        while True:
            now = time.time()
            # EEG
            sample, ts = eeg_inlet.pull_sample(timeout=0.01)
            if sample is not None:
                row = {'lsl_timestamp': ts, 'local_time': now}
                for i, val in enumerate(sample):
                    row[f'ch{i+1}'] = val
                row['marker'] = ''
                pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
                # Print apenas a cada print_interval
                if now - last_print_time > print_interval:
                    print(f"EEG t_LSL={ts:.4f} | t_local={now:.4f} | {sample}")
                    last_print_time = now

            # Marker
            while True:
                marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                if marker is not None:
                    now_marker = time.time()
                    row = {'lsl_timestamp': marker_ts, 'local_time': now_marker}
                    for i in range(n_chans):
                        row[f'ch{i+1}'] = np.nan
                    row['marker'] = marker[0]
                    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
                    if now_marker - last_print_time > print_interval:
                        print(f"MARCADOR t_LSL={marker_ts:.4f} | t_local={now_marker:.4f} | {marker[0]}")
                        last_print_time = now_marker
                else:
                    break
    except KeyboardInterrupt:
        print("Finalizando recepção.")
        print(f"Os dados estão sempre atualizados em {csv_path}")

if __name__ == "__main__":
    main()
