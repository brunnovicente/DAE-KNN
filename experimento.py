import sys
import pandas as pd
import numpy as np
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values


X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)


dados = pd.DataFrame(X)
dados['classe'] = Y
rotulados = [50 , 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]


for k in np.arange(10):
    sys.stdout.write('Execução '+str(k+1))
    X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
    
    for r, p in enumerate(porcentagem):
        
        resultadoT = pd.DataFrame()
        resultadoI = pd.DataFrame()
        
        """ PROCESSO TRANSDUTIVO """
        for j in np.arange(10):
            sys.stdout.write('\n---> ROTULADOS '+str(rotulados[r])+' - '+str(r))
            L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
            DaeKnn = DAEKNN(np.size(np.unique(y_train)), np.size(L, axis=1), 5)
            preditas = DaeKnn.fit(L, U, y)
            resultadoT['exe'+str(j+1)] = preditas
            resultadoT['y'+str(j+1)] = yu
            p += 1
            por = (p / 1000)*100
            
            X_treino = pd.DataFrame(L)
            X_treino = pd.concat([X_treino, pd.DataFrame(U)])
            Y_treino = pd.concat([pd.Series(y), pd.Series(preditas)])
            
            preditasI = DaeKnn.fit(X_treino.values, X_test, Y_treino.values)
            resultadoI['exe' + str(j+1)] = y_test
            resultadoI['y' + str(j+1)] = preditasI
            
            
        resultadoT.to_csv('resultados/resultado_T'+str(rotulados[r])+''+str(r)+'.csv', index=False) #Transdutivo
        resultadoI.to_csv('resultados/resultado_I'+str(rotulados[r])+''+str(r)+'.csv', index=False) #Indutivo

