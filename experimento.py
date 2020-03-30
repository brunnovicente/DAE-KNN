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
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time

sca = MinMaxScaler()
base = 'mnist'
caminho = 'D:/Drive UFRN/bases/'
dados = pd.read_csv(caminho + base + '.csv')

X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

dados = pd.DataFrame(X)
dados['classe'] = Y
#rotulados = [50 , 100, 150, 200, 250, 300]
#porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

for r, p in enumerate(porcentagem):
    
    resultadoT = pd.DataFrame()
    resultadoI = pd.DataFrame()
    acuraciai = []
    acuraciat = []
    kappai = []
    kappat = []
        
    for k in np.arange(10):
        print('Teste: '+str(rotulados[r])+' - '+str(k+1))
        inicio = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
        
                
        """ PROCESSO TRANSDUTIVO """
        L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
        DaeKnn = DAEKNN(np.size(np.unique(y_train)), np.size(L, axis=1), 5)
        preditas = DaeKnn.fit(L, U, y)
        
        acuraciat.append(accuracy_score(yu, preditas))
        kappat.append(cohen_kappa_score(yu, preditas))
        
        acuraciai.append(accuracy_score(y_test, DaeKnn.predizer(L, X_test, y)))
        kappai.append(cohen_kappa_score(y_test, DaeKnn.predizer(L, X_test, y)))
        
        """ Teste de outros algoritmos """

        fim = time.time()
        tempo = np.round((fim - inicio)/60,2)
        print('........ Tempo: '+str(tempo)+' minutos.')
    
    resultado['R'] = rotulados
    resultado['AT'] = acuraciat
    resultado['KT'] = kappat
    resultado['KI'] = acuraciai
    resultado['KI'] = kappai
                      
