import sys
import pandas as pd
import numpy as np
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sca = MinMaxScaler()

dados = pd.read_csv('c:/basedados/sementes.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y

p = 0
for i in np.arange(100)+1:
    sys.stdout.write('Porcentagem %.2f %%' %p)
    resultado = pd.DataFrame()
    for j in np.arange(10):
        L, U, y, yu = train_test_split(X,Y, train_size=0.1, test_size=0.9, stratify=Y)
        DaeKnn = DAEKNN(np.size(np.unique(Y)), np.size(L, axis=1), i)
        preditas = DaeKnn.fit(L, U, y)
        resultado['exe'+str(j+1)] = preditas
        p += 1
        por = (p / 1000)*100
        sys.stdout.write('\nPorcentagem %.2f %%' %por)
    resultado.to_csv('resultados/resultado_k'+str(i)+'.csv', index=False)
