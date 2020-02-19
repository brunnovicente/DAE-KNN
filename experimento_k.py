import sys
import pandas as pd
import numpy as np
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y
tamanho = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0042, 0.0084, 0.0126, 0.01675, 0.0209, 0.0251]

for i, p in enumerate(porcentagem):
    for k in np.arange(100)+1:
        resultado = pd.DataFrame()
        print('Execução '+str(tamanho[i])+' - '+str(k))
        inicio = time.time()
        for j in np.arange(1):
            L, U, y, yu = train_test_split(X,Y, train_size = p, test_size = 1.0 - p, stratify = Y)
            DaeKnn = DAEKNN(np.size(np.unique(Y)), np.size(L, axis=1), k)
            preditas = DaeKnn.fit(L, U, y)
            resultado['exe'+str(j+1)] = preditas
            resultado['y'+str(j+1)] = yu
        fim = time.time()
        tempo = np.round((fim - inicio)/60, 2)
        print('................ Tempo '+str(tempo)+' minutos.')
        resultado.to_csv('resultados/resultado_'+str(p)+'k'+str(i)+'.csv', index=False)
    break
