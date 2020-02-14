import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

dados= pd.read_csv('resultados/resultado_k50.csv')

acuracia = []
for i in np.arange(10)+1:
    y_des = dados['y'+str(i)]
    y_pre = dados['exe'+str(i)]
    acuracia.append(accuracy_score(y_pre, y_des))