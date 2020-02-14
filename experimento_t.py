import sys
import pandas as pd
import numpy as np
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sca = MinMaxScaler()

dados = pd.read_csv('c:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y

T = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

p = 0
for t in T:
    sys.stdout.write('\nPorcentagem %.2f %%' %((p/1000)*100))
    resultado = pd.DataFrame()
    
    for j in np.arange(10):
        L, U, y, yu = train_test_split(X,Y, train_size=0.1, test_size=0.9, stratify=Y)
        DaeKnn = DAEKNN(np.size(np.unique(Y)), np.size(L, axis=1), 5)
        preditas = DaeKnn.fit(L, U, y)
        resultado['exe'+str(j+1)] = preditas
        resultado['y'+str(j+1)] = yu
        p += 1
        por = (p / 1000)*100
        sys.stdout.write('\nPorcentagem %.2f %%' %por)
    resultado.to_csv('resultados/resultado_k'+str(i)+'.csv', index=False)