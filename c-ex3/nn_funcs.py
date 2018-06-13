from scipy.special import expit
import numpy as np

# feed the X matrix through the network defined an array of theta matrices
def feed_forward(X, thetas):
	a = X
	for theta in thetas:
		a = np.hstack((np.ones((len(a), 1)), a))
		z = a.dot(theta.T)
		a = expit(z)
	return a

# calculate the accuracy of the model
def calc_acc(hs, ys):
	preds = np.argmax(hs, axis=1)
	preds += 1 # add one since Matlab is stupid and indexes arrays at 1 and therefore
	# the y array represents 0 as 10
	return np.sum(preds == ys[:,0]) / float(len(preds))
