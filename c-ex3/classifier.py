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
		y, n = self.y, self.target
		threshold = lambda i: 1 if i == n else 0
		v_thresh = vectorize(threshold)
		self.y = v_thresh(y)

	# return the cost (unregularized) of the model
	def cost(self):
		return -average(
			self.y * log(self.h) +
			(1 - self.y) * log(1 - self.h))

	# return a (n,1) array that is the gradient of the current cost function
	def grad(self):
		g = self.X.T.dot(self.h - self.y) / self.m
		reg_vec = self.theta
		reg_vec[0,0] = 0
		g += reg_vec / self.m
		return g

	# computes the hessian of the current model (n x n)
	def hessian(self):
		hh_1 = diag(self.h[:,0]).dot(diag((1 - self.h)[:,0]))
		h = self.X.T.dot(hh_1).dot(self.X) / self.m
		reg_mat = identity(self.n)
		reg_mat[0,0] = 0
		h += reg_mat / self.m
		return h

	# update the current weights according to newton's method
	def update(self):
		g, hess = self.grad(), self.hessian()
		delta = 0

		# if the hessian is singular, than solve raises an exception
		# and pinv must be used
		try:
			delta = solve(hess, g)
		except LinAlgError as err:
			delta = pinv(hess).dot(g)
		self.theta -= delta

	# learn the weights to best classify the data
	def train(self, epsilon=1e-10, max_iter=1000):
		self.theta = zeros((self.n, 1))

		# when scipy.optimize.minimize is used, create two nested functions:
		#	f - the cost of the model
		#	fgrad - the gradient of the model
		# scipy takes care of the rest of the optimization

		# given a theta vector, compute the cost
		def f(theta, *args):
			self.theta = theta.reshape((self.n, 1))
			self.h = expit((self.X).dot(self.theta))
			return self.cost()

		# given a theta vector, compute the gradient of the cost
		def fgrad(theta, *args):
			self.theta = theta.reshape((self.n, 1))
			self.h = expit((self.X).dot(self.theta))
			return self.grad().flatten()

		self.theta = minimize(f, self.theta, jac=fgrad, method='CG').x 
