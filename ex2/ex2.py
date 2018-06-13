import numpy as np
import plotly.offline as pltly
from plotly.graph_objs import Scatter, Layout, Surface, Scatter3d, Contour

import argparse
parser = argparse.ArgumentParser(description='Toy linear regression.')
parser.add_argument('--plot', dest='shd_plt', action='store_true', help='Whether or not the data should be plotted')
args = parser.parse_args()

# load and visualize the data
def load_data():
	return np.genfromtxt('data/ex2x.dat'), np.genfromtxt('data/ex2y.dat')

def plot_data(x, y, theta=None, filename='tmp.html'):
	pts = Scatter(x=x, y=y, mode='markers')
	data = [pts]

	if not theta is None:
		xs = [0, max(x)]
		ys = [
			theta[0],
			theta[0] + max(x)*theta[1]]
		linreg_line = Scatter(x=xs, y=ys, mode='lines')
		data.append(linreg_line)

	layout = Layout(
		title='Age vs. Height',
		xaxis=dict(title='Age (years)'),
		yaxis=dict(title='Height (meters)'))
	pltly.plot({
		"data": data,
		"layout": layout}, filename=filename)

def plot_cost(J_surf, theta, filename="cost.html"):
	data = [Contour(z=J_surf, x=np.linspace(-3,3,100).tolist(), y=np.linspace(-3,3,100).tolist())]
	data.append(Scatter(
		x=[theta[0]], y=[theta[1]], mode='markers', marker=dict(
			size=12.5, line=dict(width=2, color='rgb(0,0,0)'))))
	layout = Layout(
		title='Model Cost',
		xaxis=dict(title='Theta 0'),
		yaxis=dict(title='Theta 1')
	)
	pltly.plot({"data": data, "layout": layout}, filename=filename)	

x, y = load_data()
if args.shd_plt:
	plot_data(x, y, filename="data.html")

# add the bias column at x
m = len(x) # the number of training examples
x = np.reshape(x, (m, 1)) # reshape x to be a matrix
x = np.hstack((np.ones((m, 1)), x)) # append a column vector of ones to the start of x

alpha = 0.07 # set the learning rate
theta = np.array([0, 0]) # initialize the parameters to 0
theta = np.reshape(theta, (len(theta), 1)).T

# compute the cost of the current hypothesis. only needed for visualization purposes 
def cost(h, y, m):
	h = h[:,0]
	y = y[:,0]
	return np.sum((h-y)**2)/m

# propagate the input data (x) through the hypothesis function
def forward_prop(theta, x):
	return np.dot(x, theta.T)

# calculate the cost given x and y and then update the parameters theta accordingly
def backward_prop(h, theta, x, y, alpha, m):
	grad = np.sum(
		np.dot((h-y).T, x),
		axis=0)
	grad /= m
	return grad, theta - alpha*grad

y = np.reshape(y, (m, 1)) # reshape y so that it is a matrix

# function to update the parameters and return the gradient
def update(x, y, theta, alpha, m):
	h = forward_prop(theta, x)
	return backward_prop(h, theta, x, y, alpha, m)

# hyperparameters for training
iter_no = 0
max_iter = 10000
tolerance = 0.0001

# update theta until it converges (the gradient is near 0)
grad, theta = update(x, y, theta, alpha, m)
while abs(np.sum(grad)) > tolerance and iter_no < max_iter:
	grad, theta = update(x, y, theta, alpha, m)
	iter_no += 1

print 'Final - theta_0: %f, theta_1: %f' % (theta[0][0], theta[0][1])
theta_vec = theta[0]
plot_data(x[:,1], y[:,0], theta=theta[0], filename="data_w_reg.html")

J_surf = np.zeros((100, 100), dtype=float)
t0_vals = np.linspace(-3, 3, 100)
t1_vals = np.linspace(-1, 1, 100)

for i, t0 in enumerate(t0_vals):
	for j, t1 in enumerate(t1_vals):
		theta = np.array([t0, t1])
		theta = np.reshape(theta, (1, len(theta)))
		J_surf[i, j] = cost(forward_prop(theta, x), y, m)

plot_cost(J_surf, theta_vec, filename="cost_surf.html")
