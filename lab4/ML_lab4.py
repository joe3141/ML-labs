from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.utils.extmath import cartesian
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np

digits = datasets.load_digits()
images = digits.images
targets = digits.target

def combine_to_tuples(arrays):
	return [tuple(i) for i in cartesian(arrays)]

def print_gs_res(grid_result):

	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	# means = grid_result.cv_results_['mean_test_score']
	# stds = grid_result.cv_results_['std_test_score']
	# params = grid_result.cv_results_['params']
	# for mean, stdev, param in zip(means, stds, params):
	#     print("%f (%f) with: %r" % (mean, stdev, param))


#plt.imshow(images[6],cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

images_flat = images.reshape(images.shape[0],-1)
norm_images = images_flat/16

X_train,X_test,Y_train,Y_test = train_test_split(norm_images,targets,test_size=0.2,random_state=4)

NN = 30
NL = 24
Lambda = 1e-02
BS = 64
ir = 0.001
pt = 0.5

l1 = [15, 30, 45] 
l2 = [12, 24, 36]
l3 = [6, 10, 12]

params = {'hidden_layer_sizes': combine_to_tuples((l1, l2, l3)) + combine_to_tuples(([30, 45, 60], l2))}

clf = MLPClassifier(
    hidden_layer_sizes=(NN,NL),activation='logistic',
    solver='adam',alpha=Lambda,batch_size=BS,
    learning_rate='invscaling',learning_rate_init=ir,power_t=pt,max_iter=5000,
    shuffle=True,random_state=1,tol=0.0001,verbose=True)


clf.fit(X_train,Y_train)
# train_scores, valid_scores = validation_curve(clf, X_train, Y_train, 'alpha', 
# 	np.logspace(-7,3,3), cv=5, scoring='accuracy')

# print("Train: ", train_scores)
# print("Valid: ", valid_scores)

# gs = GridSearchCV(clf, param_grid=params)
# gs_res = gs.fit(X_train, Y_train)
# print_gs_res(gs_res)

predictions = clf.predict(X_test)

print(metrics.accuracy_score(Y_test,predictions))

