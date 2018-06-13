from numpy import *
from scipy.io import loadmat
from scipy.special import expit

import os.path

from classifier import LogRegMulti

if __name__ == '__main__':
	mat = loadmat('data/data.mat')
	X, y = mat['X'], mat['y']

	# stupid Matlab people use 1-base indexing and so represented 0 as 10. let's change that
	for i, row in enumerate(y):
		if row[0] == 10:
			y[i] = [0]

	k = 10
	thetas = zeros((k, X.shape[1]+1))

	# if there is a file in the current directory named 'weights.txt' then
	# load the weights from that file. Otherwise, train the classifiers
	# and save the weights
	if not os.path.isfile('weights.txt'):
		for i in range(k):
			# create a LogRegMulti class objec5 to take care of the classification
			print 'Training classifier for %d' % i
			logreg = LogRegMulti(X, y, i)	
			logreg.train()
			thetas[i] = logreg.theta.T
		savetxt('weights.txt', thetas)
	else:
		thetas = genfromtxt('weights.txt')

""" *** TODO *** """
# calculate the accuracy of the model and print it
