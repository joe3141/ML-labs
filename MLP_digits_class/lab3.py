import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

"""
df = pd.read_csv('USA_Housing.csv')
# print(df.info())

X = df[['Avg. Area Income', "Avg. Area House Age", 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
Y = df['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# print(X_train.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)

# print(lm.coef_)
# print(lm.intercept_)

hypothesis = lm.predict(X_test)
# print(metrics.mean_absolute_error(hypothesis, Y_test))
"""




digits = datasets.load_digits()
images = digits.images
targets = digits.target

# plt.imshow(images[0], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

images_flat = images.reshape(images.shape[0], -1)
norm_images = images_flat / images_flat.max()
# print(norm_images[0])

X = norm_images
Y = targets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

clf = MLPClassifier(solver='sgd', alpha=0.001, hidden_layer_sizes=[64,], max_iter=1000, random_state=101)

clf.fit(X_train, Y_train)

hypothesis = clf.predict(X_test)
print(metrics.mean_absolute_error(hypothesis, Y_test))

print(hypothesis[:10])
print(Y_test[:10])