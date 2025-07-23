import csv
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
import pandas as pd

def main():
    # Lê parâmetros dos blocos esperados
    params = pd.read_csv('parametros_blocos.csv')
    expected_markers = set(params['marcador'].tolist())
    received_markers = set()

    print("Buscando streams de EEG e Markers (pelo 'type')...")

    eeg_streams = resolve_byprop('type', 'EEG')
    if not eeg_streams:
        print("Nenhum stream EEG encontrado!")
        return
    eeg_inlet = StreamInlet(eeg_streams[-1])

    marker_streams = resolve_byprop('type', 'Markers')
    if not marker_streams:
        print("Nenhum stream Markers encontrado!")
        return
    marker_inlet = StreamInlet(marker_streams[-1])

    n_chans = eeg_inlet.channel_count

    csv_path = 'dados_recebidos_teste_params.csv'
    ch_names = [f'ch{i+1}' for i in range(n_chans)]
    columns = [
        'lsl_timestamp', 'lsl_timestamp_corr', 'local_time'
    ] + ch_names + ['marker', 'loop_time_ms']

    print(f"\nGravação iniciada: {csv_path}")
    print("Recebendo dados... (vai finalizar quando receber todos os marcadores esperados)\n")

    primeira_amostra = True

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        while True:
            tic = time.perf_counter()
            now = time.time()
            sample, ts = eeg_inlet.pull_sample(timeout=0.01)

            marker_rows = []
            while True:
                marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                if marker is None:
                    break
                now_marker = time.time()
                m_str = marker[0]
                marker_ts_corr = marker_ts + marker_inlet.time_correction()
                row = [marker_ts, marker_ts_corr, now_marker] + [np.nan]*n_chans + [m_str, None]
                marker_rows.append(row)
                print(f"MARCADOR: {m_str} | t_LSL={marker_ts:.4f} | corr={marker_ts_corr:.4f}")

                received_markers.add(m_str)

            eeg_row = None
            if sample is not None:
                ts_corr = ts + eeg_inlet.time_correction()
                eeg_row = [ts, ts_corr, now] + list(sample) + ['', None]

                if primeira_amostra:
                    print(f"EEG conectado: {n_chans} canais")
                    primeira_amostra = False

            toc = time.perf_counter()
            loop_time_ms = 1000 * (toc - tic)

            for row in marker_rows:
                row[-1] = loop_time_ms
                writer.writerow(row)
            if eeg_row is not None:
                eeg_row[-1] = loop_time_ms
                writer.writerow(eeg_row)

            if expected_markers.issubset(received_markers):
                print("\nTodos os marcadores esperados foram recebidos!")
                break

    print(f"\nRecepção finalizada. Dados salvos em {csv_path}")

if __name__ == "__main__":
    main()
