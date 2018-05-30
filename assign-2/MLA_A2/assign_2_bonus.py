from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import cartesian
from sklearn import metrics
import numpy as np
import pandas as pd
from utils import *

def chunk(n, lst):
	n = min(n, len(lst)-1)
	return [lst[i:i+n] for i in range(len(lst) - n+1)]

ngp = pd.read_csv('ngp.csv')
# print(ngp.info())
ngp = ngp.dropna()
ngp = ngp[:-1] # Drop a row so it would be divisible by 5

prices = np.array([i[::-1] for i in chunk(5, np.array([i[0] for i in ngp[['price']].values], dtype=np.float64))[::-1]])

X = prices[:, range(4)]
y = prices[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train , scaler = scale(X_train)

NN = 15
NL = 12
Lambda = 1e-2
BS = 64
ir = 1e-3
pt = 0.5
act = 'relu' 

t2 = (45, 36, 10)

params = {'hidden_layer_sizes': combine_to_tuples((np.arange(30,60,4), np.arange(15, 45, 4), np.arange(12, 36, 2))) + combine_to_tuples((np.arange(30,60,4), np.arange(12, 36, 2))) \
+ [i for i in np.arange(15,50,5)], 'batch_size': [32, 64, 128, 256], 'alpha': np.logspace(-7,3,3)}

# Best: 0.957570 using {'hidden_layer_sizes': (42, 14), 'batch_size': 32, 'alpha': 0.01}
# 0.960154 using {'hidden_layer_sizes': (54, 43, 26), 'batch_size': 128, 'alpha': 0.01}


# reg = MLPRegressor(
#     hidden_layer_sizes=t2,activation=act,
#     solver='adam',alpha=Lambda,batch_size=BS,
#     learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
#     shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=False)

reg = MLPRegressor(
    hidden_layer_sizes=(42, 14),activation=act,
    solver='adam',alpha=Lambda,batch_size=32,
    learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
    shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=True)

# reg = MLPRegressor(
#     hidden_layer_sizes=(54, 43, 26),activation=act,
#     solver='adam',alpha=Lambda,batch_size=128,
#     learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
#     shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=False)

reg.fit(X_train, y_train)
final_results(reg, scaler.transform(X_test), y_test)

# rs = RandomizedSearchCV(reg, param_distributions=params)
# rs_res = rs.fit(X_train, y_train)
# print_gs_res(rs_res)