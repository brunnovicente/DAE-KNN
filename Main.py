import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from DEC import DeepEmbeddingClustering
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score
import keras.backend as K
from scipy.stats import t
from scipy.stats import norm
from SKNN import Semi_Supervised_KNN
from DAE_KNN import DAEKNN

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('c:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y


L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

DK = DAEKNN(3, np.size(L, axis=1))
DK.reducaoZ(U)

PL = DK.encoder.predict(L)
PU = DK.encoder.predict(U)

knn = Semi_Supervised_KNN()
rotulos = knn.classificar(PL, PU, y, k=5)
acuracia = accuracy_score(yu, rotulos)