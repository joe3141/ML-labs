import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.decomposition import PCA

from utils import *
import itertools

X_train = read_file("train/X_train.txt")
y_train = np.ravel(read_file("train/y_train.txt"))

X_test = read_file("test/X_test.txt")
y_test = np.ravel(read_file("test/y_test.txt"))

params = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 1e-4]}

svm = SVC(C=1000, gamma=1e-4, random_state=101)


# plot_validation(X_train, y_train, svm, params['C'], 'C', 'SVM Validation Curve Under C Parameter')
# plot_validation(X_train, y_train, svm, params['gamma'], 'gamma', 'SVM Validation Curve Under gamma Parameter')


mlp = MLPClassifier(
    hidden_layer_sizes=(400),activation='relu',
    solver='adam',alpha=1e-2,batch_size=32,
    learning_rate='invscaling',learning_rate_init=1e-3,power_t=0.5,max_iter=5000,
    shuffle=True,random_state=101,tol=1e-4)

# plot_validation(X_train, y_train, mlp,  np.logspace(-7,3,3), 'alpha', 'MLP Validation Curve Under alpha Parameter')

# The last parameter encodes the parameter range into numbers from 0-29, since I was not able to plot sequences
# on the x_axis.
# plot_validation(X_train, y_train, mlp, \
# 	[x for x in itertools.product((300,400,500,600,700),repeat=1)] + \
# 	[x for x in itertools.product((300,400,500,600,700),repeat=2)], \
# 	'hidden_layer_sizes', 'MLP Validation Curve Under hidden_layer_sizes Parameter',\
# 	 param_range = np.arange(1, 30, 1))




# print("SVM Report:")
# classify(X_train, y_train, X_test, y_test, svm)


# print("MLP Report:")
# classify(X_train, y_train, X_test, y_test, mlp)

l = [5, 50, 200, 500]

for i in l:
	pca = PCA(n_components=i, random_state=101)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca  = pca.transform(X_test)

	print("SVM report for %d components:" % i)
	classify(X_train_pca, y_train, X_test_pca, y_test, svm)


	print("MLP report for %d components:" % i)
	classify(X_train_pca, y_train, X_test_pca, y_test, mlp)
