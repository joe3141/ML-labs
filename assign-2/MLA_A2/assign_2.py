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

ts_8_2 = pd.read_csv('timesereis_8_2.csv')
# print(ts_8_2.info())

X = ts_8_2[[str(i) for i in range(8)]]
y = ts_8_2[['8', '9']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train , scaler = scale(X_train)

# NN = 15
# NL = 12
Lambda = 1e-2
BS = 64
ir = 1e-3
pt = 0.5
act = 'relu' 

l1 = [15, 30, 45] 
l2 = [12, 24, 36]
l3 = [6, 10, 12]

# 30 12 12 r2 = 0.9611 ms = 0.166
# 15 24 10
# 45 36 10 r2 = .9608 ms = 0.167

t1 = (30, 12, 12)
t2 = (45, 36, 10)
t3 = (15, 24, 10)
t4 = (30, 36, 6) # r2 = 0.9602 ms = 0.1699, early_stopping = true
t5 = (60, 36)
t6 = (30, 43, 22)

# Best: 0.927768 using {'alpha': 9.9999999999999995e-08, 'batch_size': 32, 'hidden_layer_sizes': (45, 36, 10)}
# grid

# params = {'hidden_layer_sizes': combine_to_tuples((l1, l2, l3)) + combine_to_tuples(([30, 45, 60], l2)) \
# + [(15), (20), (30), (50)]}

# params = {'hidden_layer_sizes': combine_to_tuples((l1, l2, l3)) + combine_to_tuples(([30, 45, 60], l2)) \
# + [(15), (20), (30), (50)], 'batch_size': [32, 64, 128, 256], 'alpha': np.logspace(-7,3,3)}

params = {'hidden_layer_sizes': combine_to_tuples((np.arange(30,60,4), np.arange(15, 45, 4), np.arange(12, 36, 2))) + combine_to_tuples((np.arange(30,60,4), np.arange(12, 36, 2))) \
+ [i for i in np.arange(15,50,5)], 'batch_size': [32, 64, 128, 256], 'alpha': np.logspace(-7,3,3)}

reg = MLPRegressor(
    hidden_layer_sizes=t2,activation=act,
    solver='adam',alpha=Lambda,batch_size=BS,
    learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
    shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=True)

# reg = MLPRegressor(
#     hidden_layer_sizes=(20),activation=act,
#     solver='adam',alpha=1e-7,batch_size=32,
#     learning_rate='invscaling',learning_rate_init=0.1,power_t=pt,max_iter=5000,
#     shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=False)

# reg = MLPRegressor(
#     hidden_layer_sizes=t2,activation=act,
#     solver='adam',alpha=1e-7,batch_size=32,
#     learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
#     shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=True)

# reg = MLPRegressor(
#     hidden_layer_sizes=(42, 14),activation=act,
#     solver='adam',alpha=Lambda,batch_size=32,
#     learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
#     shuffle=True,random_state=1,tol=1e-4,verbose=True, early_stopping=True)




reg.fit(X_train, y_train)

# train_scores, valid_scores = validation_curve(reg, X_train, y_train, 'alpha', 
# 	np.logspace(-7,3,3), scoring='r2')

# print("Train: ", train_scores)
# print("Valid: ", valid_scores)

# train_scores, valid_scores = validation_curve(reg, X_train, y_train, 'batch_size', 
# 	[32, 64, 128, 256, 512], cv=5, scoring='r2')

# print("Train: ", train_scores)
# print("Valid: ", valid_scores)

# gs = GridSearchCV(reg, param_grid=params)
# gs_res = gs.fit(X_train, y_train)
# print_gs_res(gs_res)

# rs = RandomizedSearchCV(reg, param_distributions=params)
# rs_res = rs.fit(X_train, y_train)
# print_gs_res(rs_res)

final_results(reg, scaler.transform(X_test), y_test)
