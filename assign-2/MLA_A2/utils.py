from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import cartesian
from sklearn import metrics

def print_gs_res(grid_result):

	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	# means = grid_result.cv_results_['mean_test_score']
	# stds = grid_result.cv_results_['std_test_score']
	# params = grid_result.cv_results_['params']
	# for mean, stdev, param in zip(means, stds, params):
	#     print("%f (%f) with: %r" % (mean, stdev, param))

def scale(X):
	scaler = StandardScaler()
	X_res = scaler.fit_transform(X)
	return X_res, scaler

def final_results(reg, X, y):
	predictions = reg.predict(X)
	print("r2:", metrics.r2_score(y,predictions))
	print("variance:", metrics.explained_variance_score(y,predictions))
	print("mean squared error:", metrics.mean_squared_error(y,predictions))


def combine_to_tuples(arrays):
	return [tuple(i) for i in cartesian(arrays)]