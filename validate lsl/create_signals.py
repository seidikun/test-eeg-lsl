# gerar_sinais.py
import numpy as np
import pandas as pd

def gerar_sinais(duration=10, fs=250, n_channels=4):
    t = np.arange(0, duration, 1/fs)
    sinais = {}
    for ch in range(n_channels):
        # Sinal simples: seno
        seno = np.sin(2 * np.pi * (5 + ch*5) * t)
        # Sinal complexo: soma de senos e ruído
        complexo = (np.sin(2*np.pi*5*t) + 
                    0.5*np.sin(2*np.pi*20*t) + 
                    0.2*np.random.randn(len(t)))
        sinais[f'ch{ch+1}_seno'] = seno
        sinais[f'ch{ch+1}_complexo'] = complexo
    sinais['timestamp'] = t
    df = pd.DataFrame(sinais)
    df.to_csv('sinais_simulados.csv', index=False)
    print('Arquivo salvo como sinais_simulados.csv')

if __name__ == "__main__":
    gerar_sinais()
