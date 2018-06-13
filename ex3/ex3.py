import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go

# plot the cost over time for multiple values of alpha
def plot_cost(Js):
    data = [
		go.Scatter(x=np.linspace(0, len(J), len(J)), y=J/10e10, name='alpha='+str(a)) 
		for J, a in zip(Js, [0.5, 0.05, 0.005])]
    layout = go.Layout(
        title='Model Cost',
        xaxis=dict(title='Iteration'),
        yaxis=dict(title='Cost (x 10^10)'))
    plot({"data": data, "layout": layout})

def plot_data(x, y, theta, filename='data_linreg.html'):
	#print theta
	#print theta[0][0], theta[1][0]
	scatter = go.Scatter(
		x=x, y=y, mode='markers')
	lx = np.min(x)
	rx = np.max(x)
	ly = theta[0][0] + lx * theta[1][0]
	ry = theta[0][0] + rx * theta[1][0]
	print lx, ly
	print ly, ry
	print x
	print y
	line = go.Scatter(x=[lx, rx], y=[ly,ry], mode='lines')
	data = [scatter, line]
	layout = go.Layout(
		title='House Price',
		xaxis=dict(title='square feet'),
		yaxis=dict(title='price'))
	plot({"data": data, "layout": layout}, filename=filename)

# compute the hypothesis
def feed_forward(theta, x):
	return np.dot(x, theta.T)

# compute the gradient for the given hypothesis
def grad(h, x, y, m):
	return np.sum(
		np.dot((h-y).T, x),
		axis=0)/m

# calculate the cost for a given hypothesis
def cost(h, y, m):
	err = h - y
	return np.dot(err.T, err)/(2*m)

x = np.genfromtxt("data/ex3x.dat", dtype=float)

# normalize the dataset
def normalize(x):
	sigma = np.std(x, axis=0)
	mu = np.mean(x, axis=0)
	x[:,0] = (x[:,0] - mu[0]) / sigma[0]
	x[:,1] = (x[:,1] - mu[1]) / sigma[1]
	return x

# add the bias column to the dataset
mu = np.mean(x, axis=0)
sigma = np.std(x, axis=0)
x_norm = np.hstack((np.ones((len(x), 1)), normalize(x)))
x = np.hstack((np.ones((len(x), 1)), x))

y = np.genfromtxt("data/ex3y.dat", dtype=float)
y = np.reshape(y, (len(y), 1))

m = x.shape[0]
n = x.shape[1]
theta = np.zeros((1, n), dtype=float)
alpha = 0.07

# computer the cost over time for a given learning rate
def cost_for_alpha(alpha, x, y, m, n):
	theta = np.zeros((1, n), dtype=float)
	num_iter = 50
	J = np.zeros((num_iter,))
	for i in range(num_iter):
		h = feed_forward(theta, x)
		J[i] = cost(h, y, m)
		theta -= alpha * grad(h, x, y, m)
	return J

def theta_from_gd(x, y, epsilon=1e-10, max_iter=10000, alpha=1):
	m, n = x.shape[0], x.shape[1]
	theta = np.zeros((1, n), dtype=float)
	num_iter = 0
	h = feed_forward(theta, x)
	lc, nc = np.inf, cost(h, y, m)
	theta -= alpha * grad(h, x, y, m)
	while lc - nc > epsilon and num_iter < max_iter:
		h = feed_forward(theta, x)
		lc, nc = np.inf, cost(h, y, m)
		theta -= alpha * grad(h, x, y, m)
		num_iter += 1
	return theta[0]

alphas = [0.5, 0.05, 0.005]
Js = list(map(
	lambda a: cost_for_alpha(a, x_norm, y, m, n), alphas))
#plot_cost(Js)

# computer theta from the normal equation
def theta_from_normal_eq(x, y):
	tmp = np.linalg.inv(np.dot(x.T, x))
	return np.dot(np.dot(tmp, x.T), y)

theta = theta_from_gd(x, y)
test_vec = np.array([1, 1650.0, 3.0])
print test_vec
print mu, sigma
test_vec[1] = (test_vec[1] - mu[0])/sigma[0]
test_vec[2] = (test_vec[2] - mu[1])/sigma[1]
print test_vec
print theta
print theta[0] + theta[1]*test_vec[1] + theta[2]*test_vec[2]
print '****'
theta = theta_from_normal_eq(x, y)
print theta

# predict the value for the given house specifications
theta_vec = theta[0]
print 'Estimated price for 1650 sq. ft. and 3 bedrooms: %f' % (theta[0]+1650*theta[1]+3*theta[2])

plot_data(x[:,1], y[:,0], theta)
