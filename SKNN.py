# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

class Semi_Supervised_KNN:
    
    def classificar(self, L, U, y, k=3):
        rotulos = np.zeros(np.size(U, axis=0), dtype=np.int64) - 1
        for i, x in enumerate(U):
            #print('Rotulando ', i)
            c = self.rotular_amostras(x, L, y, k)
            rotulos[i] = c
        
        """ Rotulando os Remanescentes """
        dados = pd.DataFrame(U)
        dados['y'] = rotulos
        
        rotulados = dados[dados['y'] != -1]
        nrot = dados[dados['y'] == -1]
        indices = nrot.index.values
        
        #Caso alguém não tenho rótulo
        if np.size(indices) != 0:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(rotulados.drop(['y'], axis=1).values, rotulados['y'].values)
            
            for i, x in enumerate(nrot.drop(['y'], axis=1).values):
                c = knn.predict([x])
                pos = indices[i]
                rotulos[pos] = c        
        
        return rotulos
    
    def rotular_amostras(self, x, L, y, k):

        """ Calculando distância da Amostra para cada elemento de L """        
        dis = []
        for xr in L:
            dis.append(distance.euclidean(x, xr))
        
        """ Descobrindo os k vizinhos mais próximos """
        rot = pd.DataFrame(L)
        rot['y'] = y
        rot['dis'] = dis
        rot = rot.sort_values(by='dis')
        vizinhos = rot.iloc[0:k,:]
        
        """ Calculando as Classes """
        classes = np.unique(y)
        P = []
        for c in classes:
            q = (vizinhos['y'] == c).sum()
            p = q / k
            P.append(p)
        classe = self.calcular_classe(P)
        
        return classe
    
    def calcular_classe(self, probabilidades):
        c = -1
        for i, p in enumerate(probabilidades):
            pr = np.round(p)
            if pr == 1.:
                c = i
                break
        return c