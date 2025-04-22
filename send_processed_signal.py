import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from pyriemann.estimation import Covariances
from pylsl import StreamInfo, StreamOutlet
import time
import os

pasta_dados   = os.path.join(os.path.dirname(__file__), 'test_data')
svm_model     = joblib.load(os.path.join(pasta_dados, 'svm_model.pkl'))
tangent_space = joblib.load(os.path.join(pasta_dados, 'tangent_space.pkl'))

# Função para criar um filtro Butterworth
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist   = 0.5 * fs
    b, a      = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return b, a

# Função para aplicar o filtro aos dados usando filtfilt
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a      = butter_bandpass(lowcut, highcut, fs, order=order)
    y         = filtfilt(b, a, data, axis=0)
    return y

# Parâmetros do arquivo
filename      = 'motor_mi_EG102_online_5c.csv'
df            = pd.read_csv(os.path.join(pasta_dados, filename), low_memory=False)

Fs            = 512  # Taxa de amostragem em Hz

# Parâmetros de eventos e colunas
colunas_dados  = ['Channel ' + str(i) for i in range(1, 17)]
dados_sensores = df[colunas_dados].values

# Parâmetros de filtragem e de aumentação de dados
params = {'win_size': 2, 'overlap': 0.1, 'fmin': 7, 'fmax': 35}
win_size = int(params['win_size'] * Fs)

# Bandas de frequência (em Hz)
bands = {'beta': (13, 30)}

# Aplicando filtros Butterworth de ordem 3 em diferentes bandas
dados_filt = {}
for band, (lowcut, highcut) in bands.items():
    dados_filt[band] = butter_bandpass_filter(dados_sensores, lowcut, highcut, Fs)

# LSL Configurações para saída dos resultados, usando as mesmas especificações iniciais
sampling_rate = 50  # Taxa de amostragem conforme o exemplo original
info = StreamInfo('Signal', 'EEG', 3, sampling_rate, 'float32', 'myuid34234')
outlet = StreamOutlet(info)
print("Enviando dados de inferência via LSL...")

# Inicializando o buffer de dados
buffer = np.zeros((win_size, len(colunas_dados)))

# Processando os dados em tempo "quase-real" usando um buffer deslizante
start_time = time.time()
inferencia_intervalo = int(Fs / sampling_rate)  # Realizar inferência a cada ~10 amostras (512/50 ≈ 10)

for current_index in range(dados_filt['beta'].shape[0]):

    # Adiciona uma nova amostra ao buffer (removendo a mais antiga)
    buffer = np.roll(buffer, shift=-1, axis=0)
    buffer[-1, :] = dados_filt['beta'][current_index, :]

    # Quando o buffer está preenchido, realizamos a inferência a cada `inferencia_intervalo` amostras
    if current_index >= win_size - 1 and current_index % inferencia_intervalo == 0:
        # Calculando a matriz de covariância da janela atual no buffer
        cov_matrix = np.cov(buffer, rowvar=False)

        # Garantindo que a matriz de covariância tenha a mesma forma da matriz de referência
        if cov_matrix.shape != tangent_space.reference_.shape:
            print("Erro: A matriz de covariância não possui a forma esperada. Pulando essa janela.")
            continue

        # Projeção para o espaço tangente
        cov_matrix_np = np.array([cov_matrix])  # Transformando em um array NumPy 3D
        cov_tangent = tangent_space.transform(cov_matrix_np)[0]  # Obtendo a representação tangente

        # Usando a função de decisão para uma saída contínua
        decision_value = svm_model.decision_function([cov_tangent])[0]
        
        # Normalizando o valor para o intervalo [-1, 1]
        output_value = np.clip(decision_value, -1.0, 1.0)

        # Criando vetor de três valores conforme solicitado
        lsl_output = [0.0, 0.0, 0.0]

        if output_value < 0:
            lsl_output[0] = output_value  # valor negativo fica na primeira posição
        elif output_value > 0:
            lsl_output[2] = output_value  # valor positivo fica na terceira posição
        # se for zero, ambos ficam zerados

        # Enviando o vetor via LSL
        outlet.push_sample(lsl_output)
        print(f"Valor decisão: {decision_value:.4f}, Vetor LSL: {lsl_output}")

    # Aguardar um intervalo de tempo (simulando um tempo real)
    time.sleep(1 / Fs)  # Tempo para simular a frequência de amostragem dos dados

print("Transmissão finalizada.")
