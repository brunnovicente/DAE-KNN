import pandas as pd
import numpy as np
import time
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

sca = MinMaxScaler()
caminho = ['D:/Drive UFRN/bases/']
bases = ['mnist', 'fashion', 'usps', 'cifar10','stl10','covtype','epilepsia','reuters']

for base in bases:
    print('BASE: '+base)
    dados = pd.read_csv(caminho + base + '.csv')
    X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
    Y = dados['classe'].values
    dados = pd.DataFrame(X)
    dados['classe'] = Y
    tamanho = [50, 100, 150, 200, 250, 300]
    porcentagem = [0.0042, 0.0084, 0.0126, 0.01675, 0.0209, 0.0251]
    
    resultadoA = pd.DataFrame()
    resultadoK = pd.DataFrame()
    
    inicio = time.time()
    for i, p in enumerate(porcentagem):
        acuracia = []
        kappa = []
        for k in np.arange(25)+1:
            print('Execução '+str(tamanho[i])+' - '+str(k))
            L, U, y, yu = train_test_split(X,Y, train_size = p, test_size = 1.0 - p, stratify = Y)
            DaeKnn = DAEKNN(np.size(np.unique(Y)), np.size(L, axis=1), k)
            preditas = DaeKnn.fit(L, U, y)
            acuracia.append(accuracy_score(yu, preditas))
            kappa.append(cohen_kappa_score(yu, preditas))
        resultadoA[str(tamanho[i])] = acuracia
        resultadoK[str(tamanho[i])] = kappa
    
    resultadoA.to_csv('resultados/resultado_k_acuracaia_'+base+'.csv', index=False)
    resultadoK.to_csv('resultados/resultado_k_kappa_'+base+'.csv', index=False)
    fim = time.time()
    tempo = np.round((fim - inicio)/60, 2)
    print('................ Tempo '+str(tempo)+' minutos.')