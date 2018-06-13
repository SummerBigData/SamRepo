import numpy as np
import plotly.offline as pltly
import plotly.graph_objs as go

import argparse
from functools import reduce

parser = argparse.ArgumentParser(description='Toy linear regression.')
parser.add_argument('--plot_data', dest='shd_plt', action='store_true', help='Whether or not the data should be plotted')
parser.add_argument('--plot_gd', dest='shd_plt_gd', action='store_true', help='Whether or not to plot the gradient descent theta values')
parser.add_argument('--plot_norm', dest='shd_plt_norm', action='store_true', help='Whether or not to plot the normal equation theta values')
args = parser.parse_args()

def load_data():
	return np.genfromtxt('data/ex5Linx.dat'), np.genfromtxt('data/ex5Liny.dat')

def plot_data(x, y, theta=None, filename='tmp.html'):
	data = [go.Scatter(x=x, y=y, mode='markers', marker=dict(size=15))]

	if not theta is None:
		xs = np.linspace(x.min(), x.max(), 100)
		ys = [reduce(
				lambda a,b: a+b, 
				[theta[i] * x_val**i for i in range(6)])
			for x_val in xs]
		linreg_line = go.Scatter(x=xs, y=ys, mode='lines')
		data.append(linreg_line) 

	layout = go.Layout(title='Exercise 5 Data')
	pltly.plot({
		"data": data,
		"layout": layout}, filename=filename)

x, y = load_data()
if args.shd_plt:
	plot_data(x, y, filename="data.html")

m = len(x)
y = y.reshape((m,1))

x = np.array([x, np.power(x,2), np.power(x,3), np.power(x,4), np.power(x,5)]).T
x = np.hstack((np.ones((m,1)), x))
n = x.shape[1]

# propagate the input data (x) through the hypothesis function
def forward_prop(theta, x):
	return x.dot(theta.T)

# calculate the cost given x and y and then update the parameters theta accordingly
def backward_prop(h, theta, x, y, lmbda, alpha, m):
	#grad = np.sum(
	#	np.dot((h-y).T, x),
	#	axis=0) + lmbda*theta
	theta_reg = theta
	theta_reg[0,0] = 0
	grad = np.dot((h-y).T, x) + lmbda*theta_reg
	grad /= m
	return grad, theta - alpha*grad

# function to update the parameters and return the gradient
def update(x, y, theta, lmbda, alpha, m):
	h = forward_prop(theta, x)
	return backward_prop(h, theta, x, y, lmbda, alpha, m)

def theta_for_lambda_gd(lmbda, x, y, m, n, alpha=0.07, epsilon=1e-5, max_iter=10000):
	theta = np.zeros((1, n), dtype=float)
	iter_no = 0

	grad, theta = update(x, y, theta, lmbda, alpha, m)
	while abs(np.sum(grad)) > epsilon and iter_no < max_iter:
		grad, theta = update(x, y, theta, lmbda, alpha, m)
		iter_no += 1
	return theta[0]

def theta_for_lambda_norm(lmbda, x, y, n, alpha=0.07):
	lmbda_mat = np.identity(n)
	lmbda_mat[0,0] = 0
	xtx = np.dot(x.T, x)
	xtx_lmbd = xtx + lmbda*lmbda_mat
	xty = (x.T).dot(y)
	return np.linalg.solve(xtx_lmbd, xty).flatten()

def plot_thetas(x, y, thetas, ttype):
    x_vals = x[:,1]
    y_vals = y[:,0]
    map(
        lambda (i,t): plot_data(
            x_vals, y_vals, theta=t,
            filename='data_linreg_' + ttype + '_' + str(i) + '.html'),
        list(enumerate(thetas)))

lambdas = [0, 1, 10]

if __name__=='__main__':
	thetas_gd = list(map(
		lambda l: theta_for_lambda_gd(l, x, y, m, n), lambdas))
	if args.shd_plt_gd:
		plot_thetas(x, y, thetas_gd, 'gd')

	thetas_norm = list(map(
		lambda l: theta_for_lambda_norm(l, x, y, n), lambdas))
	if args.shd_plt_norm:
		plot_thetas(x, y, thetas_norm, 'norm')

	for i, (tg, tn) in enumerate(zip(thetas_gd, thetas_norm)):
		print 'lambda: %d' % lambdas[i]
		print tg
		print tn
		print '***'
