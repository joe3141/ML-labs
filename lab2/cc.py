import numpy as np
import pandas as pd
from perceptron import Perceptron


cc = pd.read_csv('m_creditcard_24650.csv')
# print(cc.info())

X = cc[['V2', 'V11']]
y = cc[['Class']].values

# Standardization
X_norm = ((X - X.mean()) / (X.std())).values

# Alpha = 0.2
p1_1 = Perceptron(X.shape[1], 1, 0.2, True)
p1_1.train(X_norm, y, 1000, 10)

# Alpha = 0.8
p1_2 = Perceptron(X.shape[1], 1, 0.8, True)
p1_2.train(X_norm, y, 1000, 10)