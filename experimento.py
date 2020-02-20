import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from DAE_KNN import DAEKNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import time

sca = MinMaxScaler()

dados = pd.read_csv('c:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values


X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)


dados = pd.DataFrame(X)
dados['classe'] = Y
#rotulados = [50 , 100, 150, 200, 250, 300]
#porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

for r, p in enumerate(porcentagem):
    
    resultadoT = pd.DataFrame()
    resultadoI = pd.DataFrame()
    resultadoMLP = pd.DataFrame()
    resultadoKNN = pd.DataFrame()
    resultadoSVM = pd.DataFrame()
    resultadoRF = pd.DataFrame()
    resultadoNB = pd.DataFrame()
    resultadoLR = pd.DataFrame()
    
    for k in np.arange(10):
        print('Teste: '+str(rotulados[r])+' - '+str(k+1))
        
        X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
        
                
        """ PROCESSO TRANSDUTIVO """
        L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
        DaeKnn = DAEKNN(np.size(np.unique(y_train)), np.size(L, axis=1), 5)
        
        inicio = time.time()
        preditas = DaeKnn.fit(L, U, y)
        resultadoT['exe'+str(k+1)] = preditas
        resultadoT['y'+str(k+1)] = yu
                    
        X_treino = pd.DataFrame(L)
        X_treino = pd.concat([X_treino, pd.DataFrame(U)])
        Y_treino = pd.concat([pd.Series(y), pd.Series(preditas)])
        
        preditasI = DaeKnn.fit(X_treino.values, X_test, Y_treino.values)
        resultadoI['exe' + str(k+1)] = y_test
        resultadoI['y' + str(k+1)] = preditasI
        
        """ Teste de outros algoritmos """
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC()
        rf = RandomForestClassifier(n_estimators=20)
        nb = GaussianNB()
        lr = LogisticRegression()
        
        mlp.fit(X_treino.values, Y_treino)
        knn.fit(X_treino.values, Y_treino)
        svm.fit(X_treino.values, Y_treino)
        rf.fit(X_treino.values, Y_treino)
        nb.fit(X_treino.values, Y_treino)
        lr.fit(X_treino.values, Y_treino)
        
        resultadoMLP['exe'+str(k+1)] = mlp.predict(X_test)
        resultadoMLP['y'+str(k+1)] = y_test
        resultadoKNN['exe'+str(k+1)] = knn.predict(X_test)
        resultadoKNN['y'+str(k+1)] = y_test
        resultadoSVM['exe'+str(k+1)] = svm.predict(X_test)
        resultadoSVM['y'+str(k+1)] = y_test
        resultadoRF['exe'+str(k+1)] = rf.predict(X_test)
        resultadoRF['y'+str(k+1)] = y_test
        resultadoNB['exe'+str(k+1)] = nb.predict(X_test)
        resultadoNB['y'+str(k+1)] = y_test
        resultadoLR['exe'+str(k+1)] = lr.predict(X_test)
        resultadoLR['y'+str(k+1)] = lr.predict(X_test)
        fim = time.time()
        tempo = np.round((fim - inicio)/60,2)
        print('........ Tempo: '+str(tempo)+' minutos.')
                    
    resultadoT.to_csv('resultados/resultado_MODELO_T_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False) 
    resultadoI.to_csv('resultados/resultado_MODELO_I_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoMLP.to_csv('resultados/resultado_MLP_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoKNN.to_csv('resultados/resultado_KNN_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoSVM.to_csv('resultados/resultado_SVM_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoRF.to_csv('resultados/resultado_RF_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoNB.to_csv('resultados/resultado_NB_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
    resultadoLR.to_csv('resultados/resultado_LR_'+str(rotulados[r])+'_'+str(k+1)+'.csv', index=False)
        
