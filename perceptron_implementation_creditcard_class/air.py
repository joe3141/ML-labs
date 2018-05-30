import numpy as np
import pandas as pd
from perceptron import Perceptron

f9 = pd.read_csv('F9.csv')[['AIR_TIME', 'DISTANCE']].dropna()
# print(f9.info())
# print(f9.head())

#Split the data randomly into training and test sets.
f9_train = f9.sample(frac=0.8, random_state=200)
f9_test = f9.drop(f9_train.index)

X_train = f9_train[['DISTANCE']]
y_train = f9_train[['AIR_TIME']].values
X_test  = f9_test[['DISTANCE']]
y_test  = f9_test[['AIR_TIME']].values

# Standardize and keep the standardization transformation for inputs outside the training set.

t_mean = X_train.mean()
t_std  = X_train.std()

X_train = ((X_train - t_mean) / t_std).values
X_test  = ((X_test  - t_mean) / t_std).values


p2 = Perceptron(X_train.shape[1])
train_rmse = p2.train(X_train, y_train, thresh=1e-9, max_iters=10000)
# train_rmse = p2.train(X_train, y_train, thresh=1e-9, max_iters=10000, epochs=10)

test_out = p2.infer(X_test)
test_rmse = np.sqrt(2*(p2._error(test_out, y_test)))

# Note for RMSE values:
# Test RMSE > Train RMSE => overfitting
# Test RMSE ~= Train RMSE => Usually a positive indicator

print("Train rmse:", train_rmse)
print("Test rmse:", test_rmse)