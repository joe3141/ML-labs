import pandas as pd
from sklearn.datasets import load_breast_cancer
from  sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

cancer = load_breast_cancer()

X = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
y = cancer['target']
# print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)

params = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 1e-4]}
grid = GridSearchCV(SVC(), param_grid=params, cv=5, refit=True, verbose=True)


grid.fit(X_train, Y_train)
preds = grid.predict(X_test)
print(grid.best_estimator_)
print(classification_report(Y_test, preds))
# print(preds)

