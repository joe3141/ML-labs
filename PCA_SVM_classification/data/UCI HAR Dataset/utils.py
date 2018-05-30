import numpy as np
from sklearn.utils.extmath import cartesian

from  sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def read_file(path):
	with open(path, "r") as f:
		return np.array([np.array(row.strip().split(), dtype=np.float32) for row in f.readlines()])

def combine_to_tuples(list):
        return [tuple(i) for i in cartesian(list)]

def classify(X_train, y_train, X_test, y_test, clf):
	clf.fit(X_train, y_train)

	preds = clf.predict(X_test)
	print(classification_report(y_test, preds))

def plot_validation(X_train, y_train, clf, param_range, param_name, title, X_vals=None):
	train_scores, test_scores = validation_curve(
    clf, X_train, y_train, param_name=param_name, param_range=param_range,
    cv=2, scoring="accuracy", n_jobs=-1)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.title(title)
	plt.xlabel("$%s$" % param_name)
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
	             color="darkorange", lw=lw)
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2,
	                 color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="navy", lw=lw)
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2,
	                 color="navy", lw=lw)
	plt.legend(loc="best")
	plt.show()
