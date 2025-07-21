from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import numpy as np
import time

def main():
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
    columns = ['lsl_timestamp', 'local_time'] + ch_names + ['marker']
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    print(f"\nGravação iniciada: {csv_path}")
    print("Recebendo dados... Pressione Ctrl+C para parar.\n")

    buffer = []
    buffer_size = 200
    primeira_amostra = True

    try:
        while True:
            now = time.time()
            sample, ts = eeg_inlet.pull_sample(timeout=0.01)

            if sample is not None:
                row = {'lsl_timestamp': ts, 'local_time': now, 'marker': ''}
                row.update(zip(ch_names, sample))
                buffer.append(row)

                if primeira_amostra:
                    print(f"EEG conectado: {n_chans} canais | Exemplo de amostra: {sample}")
                    primeira_amostra = False

            while True:
                marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                if marker is None:
                    break

                now_marker = time.time()
                row = {'lsl_timestamp': marker_ts, 'local_time': now_marker, 'marker': marker[0]}
                row.update({ch: np.nan for ch in ch_names})
                buffer.append(row)
                print(f"MARCADOR: {marker[0]} | t_LSL={marker_ts:.4f}")

            if len(buffer) >= buffer_size:
                pd.DataFrame(buffer).to_csv(csv_path, mode='a', header=False, index=False)
                buffer.clear()

    except KeyboardInterrupt:
        print("\nFinalizando recepção.")
        if buffer:
            pd.DataFrame(buffer).to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Dados salvos em {csv_path}")

if __name__ == "__main__":
    main()
