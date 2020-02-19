from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from SKNN import Semi_Supervised_KNN
import numpy as np

class DAEKNN:
    
    def __init__(self, dim, entrada, k):
        self.k = k
        self.entrada = entrada
        self.dim = dim
        
        input_img = Input((self.entrada,))
        #encoded = Dense(50, activation='relu')(input_img)
        #drop = Dropout(0.2)(encoded)
        encoded = Dense(10, activation='relu')(input_img)
        #drop = Dropout(0.2)(encoded)
        #encoded = Dense(100, activation='relu')(drop)
        
        Z = Dense(self.dim, activation='relu')(encoded)
        
        decoded = Dense(10, activation='relu')(Z)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(50, activation='relu')(drop)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(250, activation='relu')(drop)
        decoded = Dense(self.entrada, activation='sigmoid')(decoded)
                        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        #self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=0.1, decay=0, momentum=0.9))
    
    def fit(self,L, U, y):
        PU = self.reducaoZ(U)     
        PL = self.encoder.predict(L)
        return self.rotulacao(PL, PU, y)
        
    def reducaoZ(self, X):
        self.autoencoder.fit(X, X, batch_size=30, epochs=50, verbose=False)
        return self.encoder.predict(X)
    
    def rotulacao(self, PL, PU, y):
        self.knn = Semi_Supervised_KNN()
        #print('........... Tamanho Rotulados: ', str(np.size(PL, axis=1)))
        self.rotulos = self.knn.classificar(PL, PU, y, k=self.k)
        return self.rotulos
    
    def predizer(self, x):
        self.knn.classificar()
        