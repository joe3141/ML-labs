import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
	"""Perceptron"""
	def __init__(self, num_of_features, num_of_outputs=1, alpha=1e-3, classifier=False):
		self.num_of_features = num_of_features
		self.num_of_outputs = num_of_outputs # Corresponds to the number of neurons also.
		self.alpha = alpha # Learning rate.
		# Initialize weight matrix with random numbers between -0.05 - 0.05:
		self.weights = np.random.rand(num_of_features+1, num_of_outputs) * (0.01-0.05)
		# self.weights = np.zeros(shape=(num_of_features+1, num_of_outputs))
		self.classifier = classifier # Is this perceptron a classifier?
	
	# Use our model to induce a hypothesis using an input matrix X.
	def infer(self, X):
		# Does the input matrix NOT include bias terms?
		if X.shape[1] == self.num_of_features or X.shape == (self.num_of_features,):
			X = self._concatBias(X) # Add it.
		elif X.shape[1] != self.num_of_features + 1: # Does not include bias and have size more than num_feats + 1
			# Raise error since it's a peculiar size of the feature dimension.
			raise ValueError('Size mismatch: Not equal to original number of features.')	
		
		# Otherwise, the input matrix already includes bias terms.
		res = np.dot(X, self.weights) # Neuron(s) output(s).

		#Activation fn. Threshold to 0 and cut to 1 if we are classifying, otherwise the identity function.
		if self.classifier:
			return np.where(res>=0, 1., 0.) 
		return res

	"""
	Train the perceptron against a set of targets y, using dataset X.
	max_iters: If the convergence test did not fire, then only spend a maximum of "max_iters" times.
	epochs: Repeat training for a number of "epochs" times.
	thresh: Used for the convergence test, to judge whether the overall change in the weights is significant.

	Note: The final weights considered are the ones with the lowest rate of error and not necessarily the last
	weight vector calculated in the last iteration. The error metric is "Mean squared error" and is found in the
	function: _error.
	"""

	def train(self, X, y, max_iters=1000, epochs = 1, thresh=1e-6):

		X = self._concatBias(X) 
		change = np.arange(X.shape[0]) # Used later for shuffling.

		tmp = self.infer(X) # Initial hypothesis.
		error = self._error(tmp, y) # Initial error value.
		best = error # Placeholder for the best error value.
		best_wts = self.weights # Placeholder for the best weights related (or resulted) to the best error value.

		curr_iters = iters = 0 # Placeholder for outputting the number of iterations taken.
		early = False # Flag for indicating whether it converged early or taken "max_iters" iterations for a certain epoch.
		all_iters = 0.0 # Total number of iterations taken over all epochs. Will be useful for calculating the
						# average number of iterations per epoch, that is useful for commentating on the learning rate.

		for e in range(epochs): # Epoch here has a different meaning than its use in sgd. However still a correct use of the word.

			for i in range(max_iters):

				# if i % 10 == 0: # Uncomment if need to trace the error change.
				# 	print(error)

				hyp = self.infer(X) # Current hypothesis.

				# Weight update rule of the perceptron learning algorithm.
				# W -= a(y-t).x
				if self.classifier: # Only difference here is scaling by the number of samples in the training set.
									# Since regression hypotheses are much larger than that of classification.
									# (Not just a vector of [0,1]) And is done to avoid overflow of calculations.
									# Also, saving new weights to compare with current one for convergence test.

					nw_wts = self.weights - ((self.alpha) * np.dot(np.transpose(X), hyp-y))
				else:
					nw_wts = self.weights - ((self.alpha/len(y)) * np.dot(np.transpose(X), hyp-y))

				# print(np.sum(np.abs(nw_wts - self.weights)/nw_wts)) # Uncomment if need to trace weight change.

				if np.sum(np.abs(nw_wts - self.weights)/nw_wts) < thresh: # Is the weight change NOT signifcant enough?
					# Store the number of iterations taken, set the early flag and break.
					curr_iters = i
					early = True
					break

				# Otherwise, update the model's weights and calculate the error.
				self.weights = nw_wts
				error = self._error(hyp, y)

				if error < best: # Is this model better than the best one so far?
					# Update the best one so far.
					best = error
					best_wts = self.weights

				# Shuffle the dataset in unison with the target set.
				np.random.shuffle(change)
				X = X[change,:]
				y = y[change,:]

			print("\nEpoch #%d" % (e+1))
			print("Best error so far:", best)

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
				print("Best RMSE so far:", np.sqrt(2*best))
			print("error 2:", best * (y.shape[0])) # Sometimes easier to read. Particularly in the CC problem.

			# Restart. Useful in cases where previous agent got stuck in a saddle area.
			self.weights = np.random.rand(self.num_of_features+1, self.num_of_outputs) * (0.01-0.05)
			# self.weights = np.zeros(shape=(self.num_of_features+1, self.num_of_outputs))
			error = self._error(self.infer(X), y)
			olderr = np.inf

		print("\nFinished with error:", best)
		self.weights = best_wts
		hyp = self.infer(X)
		rmse = 0

		if self.classifier:
			print("With accuracy:", self._accuracy(hyp, y))
		else:
			rmse = np.sqrt(2*best) # RMSE isn't scaled by 0.5 according to the references I refered to.
			print("RMSE:", rmse)


		print("error 2:", best * (y.shape[0]))

		print("Avg number of iterations:", (all_iters/epochs))
		
		if self.classifier:
			self._plot(X, y, hyp)
		else:
			return rmse

	# Concatenate the bias terms on an arbitrary feature dataset.
	def _concatBias(self, X):
		return np.concatenate((X,-np.ones((X.shape[0],1))),axis=1)

	# Mean squared error measurement.
	def _error(self, hyp, y):
		return 0.5*(1.0/y.shape[0])*np.sum(np.power(y-hyp, 2))

	# Classification accuracy
	def _accuracy(self, hyp, y):
		return np.sum(hyp==y)/y.shape[0]

	# Plot routine only useful for plotting the decision boundary in the CC problem.
	def _plot(self, X, y, hyp):

		xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = self.infer(grid).reshape(xx.shape)

		f, ax = plt.subplots(figsize=(8, 6))
		ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

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