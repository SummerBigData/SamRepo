from numpy import *
from numpy.linalg import solve, pinv, LinAlgError
from scipy.special import expit
from scipy.optimize import minimize

# class for implementing a logistic regression classifier for the given <target> number
class LogRegMulti:
	def __init__(self, X, y, target):
		X = hstack((ones((X.shape[0],1)), X)) # append the bias column
		self.X = X
		self.m, self.n = X.shape[0], X.shape[1]

		self.y = y
		self.target = target
		self.prepro() # 

	# convert the label array to 0 if it isn't the target number and 1 if it is
	def prepro(self):
""" *** TODO *** """
# write code to preprocess the y array for this classifier

	# return the cost (unregularized) of the model
	def cost(self):
""" *** TODO *** """
# implement the cost function

	# return a (n,1) array that is the gradient of the current cost function
	def grad(self):
""" *** TODO *** """
# implement the gradient function
# return a vector of length n

	# computes the hessian of the current model (n x n)
	def hessian(self):
""" *** TODO *** """
# implement the hessian function
# return a n by n matrix

	# update the current weights according to newton's method
	def update(self):
""" *** TODO *** """
# when implementing personally, update theta according to Newton's method

	# learn the weights to best classify the data
	def train(self, epsilon=1e-10, max_iter=1000):
		self.theta = zeros((self.n, 1))

""" *** TODO *** """
# learn the theta vector either with a personal implementation or scipy
