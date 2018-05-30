import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

# a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]],dtype=float)
# b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]],dtype=float)
# c = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=float)

# p = Perceptron(2, 1, 0.25, True)
# p.train(a[:, :2], a[:, 2:], 100)
# print(p.infer(a[:, :2]))



# cc = pd.read_csv('m_creditcard_24650.csv')
# # print(cc.info())

# X = cc[['V2', 'V11']]
# y = cc[['Class']].values

# X_norm = ((X - X.mean()) / (X.std())).values

# p1_1 = Perceptron(X.shape[1], 1, 0.8, True)
# p1_1.train(X_norm, y, 1000, 10)


f9 = pd.read_csv('F9.csv')[['AIR_TIME', 'DISTANCE']].dropna()
# print(f9.info())
# print(f9.head())

f9_train = f9.sample(frac=0.8, random_state=200)
f9_test = f9.drop(f9_train.index)

X_train = f9_train[['DISTANCE']]
y_train = f9_train[['AIR_TIME']].values
X_test  = f9_test[['DISTANCE']]
y_test  = f9_test[['AIR_TIME']].values

t_mean = X_train.mean()
t_std  = X_train.std()

X_train = ((X_train - t_mean) / t_std).values
X_test  = ((X_test  - t_mean) / t_std).values


p2 = Perceptron(X_train.shape[1])
train_rmse = p2.train(X_train, y_train, thresh=1e-9, max_iters=10000)

test_out = p2.infer(X_test)
test_rmse = np.sqrt(2*(p2._error(test_out, y_test)))

print("Train rmse:", train_rmse)
print("Test rmse:", test_rmse)