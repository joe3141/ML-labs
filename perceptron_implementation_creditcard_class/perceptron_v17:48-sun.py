import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
	return np.reciprocal(1.0 + np.exp(-X))

class Perceptron(object):
	"""Perceptron"""
	def __init__(self, num_of_features, num_of_outputs=1, alpha=1e-3, classifier=False):
		self.num_of_features = num_of_features
		self.num_of_outputs = num_of_outputs # Corresponds to the number of neurons also.
		self.alpha = alpha
		# Random numbers between -0.05 - 0.05
		self.weights = np.random.rand(num_of_features+1, num_of_outputs) * (0.01-0.05)
		self.classifier = classifier 
	
	def infer(self, X):
		if X.shape[1] == self.num_of_features or X.shape == (self.num_of_features,):
			X = self._concatBias(X)
		elif X.shape[1] != self.num_of_features + 1:
			raise ValueError('Size mismatch: Not equal to original number of features.')	

		res = np.dot(X, self.weights)

		if self.classifier:
			return np.where(res>=0, 1., 0.) 
		return res

	def train(self, X, y, max_iters=1000, epochs = 1, thresh=1e-6):

		X = self._concatBias(X)
		change = np.arange(X.shape[0])

		tmp = self.infer(X)
		error = (self._error(X, tmp, y), self._error2(tmp, y))
		olderr = (np.inf,np.inf)
		best = error
		best_wts = self.weights

		curr_iters = iters = 0
		early = False
		all_iters = 0.0

		for e in range(epochs): # Epoch here has a different meaning than its use in sgd. However still a correct use of the word.

			for i in range(max_iters):

				if i % 10 == 0:
					print(error[0])

				# print(error[0])

				# if error[0] == 0.0 or np.abs(error[0]-olderr[0])/error[0] <= thresh and i > 5:
				# 	# if error[0] != 0:
				# 	# 	print("diff", np.abs(error[0]-olderr[0])/error[0])
				# 	# else:
				# 	# 	print("diff", 0.0)
					# curr_iters = i
					# early = True
					# break

				hyp = self.infer(X)
				# self.weights -= self.alpha * np.dot(np.transpose(X), hyp-y)
				self.weights -= self.alpha/len(y) * np.dot(np.transpose(X), hyp-y)


				olderr = error
				error = (self._error(X, hyp, y), self._error2(hyp, y))

				if error[0] < best[0] and error[1] < best[1]:
					best = error
					best_wts = self.weights

				np.random.shuffle(change)
				X = X[change,:]
				y = y[change,:]

			print("\nEpoch #%d" % (e+1))
			print("Best error so far:", best[0])

			if early:
				iters = curr_iters
			else:
				iters = max_iters

			all_iters += iters

			print("Epoch finished After %d iterations." % iters)
			self.weights = best_wts
			hyp = self.infer(X)

			if self.classifier:
				print("Best accuracy so far:", self._accuracy(hyp, y))
			else:
				print("Best RMSE so far:", np.sqrt(2*best[0]))
			print("error 2:", self._error2(hyp, y))

			# Restart. Useful in cases where previous agent got stuck in a saddle point.
			self.weights = np.random.rand(self.num_of_features+1, self.num_of_outputs) * (0.01-0.05)
			error = self._error(X, self.infer(X), y), self._error2(hyp, y)
			olderr = np.inf, np.inf

		print("\nFinished with error:", best[0])
		self.weights = best_wts
		hyp = self.infer(X)
		rmse = 0

		if self.classifier:
			print("With accuracy:", self._accuracy(hyp, y))
		else:
			rmse = np.sqrt(2*best[0])
			print("RMSE:", rmse)


		print("error 2:", self._error2(hyp, y))

		print("Avg number of iterations:", (all_iters/epochs))
		
		if self.classifier:
			self._plot(X, y, hyp)
		else:
			return rmse


	def _concatBias(self, X):
		return np.concatenate((X,-np.ones((X.shape[0],1))),axis=1)

	def _error2(self,hyp, y):
		return 0.5*np.sum(np.power(y-hyp, 2)) # Easier to read, in some cases.

	def _error(self, X, hyp, y):
		if self.classifier == True:
			res = sigmoid(np.dot(X, (self.weights/(np.linalg.norm(self.weights)))))
			return (-1/y.shape[0]) * np.sum((y*np.log(res)) + ((1-y)*np.log(1-res))) # Cross entropy
		else:
			return 0.5*(1.0/y.shape[0])*np.sum(np.power(y-hyp, 2)) # mean squared


	def _accuracy(self, hyp, y):
		return np.sum(hyp==y)/y.shape[0]

	def _plot(self, X, y, hyp):

		xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = self.infer(grid).reshape(xx.shape)

		f, ax = plt.subplots(figsize=(8, 6))
		ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

		# ax.scatter(X[:,0], X[:, 1], c=y[:], s=50,
		#            cmap="RdBu", vmin=-.2, vmax=1.2,
		#            edgecolor="white", linewidth=1)

		ax.scatter(X[:,0], X[:, 1], c=y[:], s=50,
		           cmap="RdBu", vmin=-.2, vmax=1.2,
		           edgecolor="white", linewidth=1)

		ax.set(aspect="equal",
		       xlim=(-5, 5), ylim=(-5, 5),
		       xlabel="$V_2$", ylabel="$V_{11}$")

		plt.show()

	# thresh = 1e-6
	# 111, 211, 91, 112, 88, 211, 106, 108, 112, 104	125.4
	# 112, 49, 101, 109, 14, 88, 33, 36, 21, 111, 36  71.0   