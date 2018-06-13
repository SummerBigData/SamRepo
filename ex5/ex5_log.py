import numpy as np
from scipy.special import expit
import scipy.optimize as opt
from plotly.offline import plot
import plotly.graph_objs as go

import argparse
parser = argparse.ArgumentParser(description='Logistic regression w/ regularization')
parser.add_argument('--plot_data', dest='shd_plt', action='store_true', help='Whether or not the data should be plotted')
parser.add_argument('--plot_db', dest='shd_plt_dec', action='store_true', help='Whether or not to plot the deciion boundary')
args = parser.parse_args()

# load and visualize the data
def load_data():
	f = open('data/ex5Logx.dat')
	lines = f.readlines()
	float_lines = list(map(
		lambda line: list(map(float, line.split(','))), lines))
	x = np.array(float_lines)
	return x, np.genfromtxt('data/ex5Logy.dat')

def plot_data(x, y, filename='tmp.html'):
	where = x[np.argwhere(y==1).flatten()]
	not_where = x[np.argwhere(y==0).flatten()]
	where_scatt = go.Scatter(
		x=where[:,0], y=where[:,1],
		name='Yes', mode='markers',
		marker=dict(
			size=10, color='rgb(255,0,0)', 
			line=dict(width=1, color='rgb(0,0,0)')))
	nwhere_scatt = go.Scatter(
		x=not_where[:,0], y=not_where[:,1],
		name='Yesn\'t', mode='markers',
		marker=dict(
			size=10, color='rgb(0,0,255)',
			line=dict(width=1, color='rgb(0,0,0)')))
	data = [where_scatt, nwhere_scatt]

	layout = go.Layout(
		title='Exercise 5 Data',
		xaxis=dict(title='U'),
		yaxis=dict(title='V'))
	plot({
		"data": data,
		"layout": layout}, filename=filename) 

def plot_db(x, y, theta, filename='db.html'):
	x_vals = x[:,1:3]
	where = x_vals[np.argwhere(y==1).flatten()]
	not_where = x_vals[np.argwhere(y==0).flatten()]
	where_scatt = go.Scatter(
		x=where[:,0], y=where[:,1],
		name='Yes', mode='markers',
		marker=dict(
			size=10, color='rgb(255,0,0)', 
			line=dict(width=1, color='rgb(0,0,0)')))
	nwhere_scatt = go.Scatter(
		x=not_where[:,0], y=not_where[:,1],
		name='Yesn\'t', mode='markers',
		marker=dict(
			size=10, color='rgb(0,0,255)',
			line=dict(width=1, color='rgb(0,0,0)')))
	data = [where_scatt, nwhere_scatt]

	u = np.linspace(-1, 1.5, 100)
	v = np.linspace(-1, 1.5, 100)
	z = np.zeros((len(u), len(v)))
	
	#for i, uval in enumerate(u):
	#	for j, vval in enumerate(v):
	#		z[i,j] = get_feature(uval, vval).T.dot(theta)
	for i, vval in enumerate(v):
		uv = np.hstack((u.reshape((100,1)), np.ones((100,1)) * vval))
		zval = np.apply_along_axis(get_feature, 1, uv)
		z[i] = zval.dot(theta.reshape((28, 1))).reshape((100,))
	z = z.T
	"""
	for i, uval in enumerate(u):
		coef = [
			sum([uval**j for k in range(j)])
			for j in reversed(range(6))]	
		coef = np.array(list(reversed(coef)))
		roots = np.poly1d(coef * theta).r
		for j, root in enumerate(roots):
			z[i,j] = root
	"""

	data.append(go.Contour(
		z=z,
		x=u, y=v,
		ncontours=1,
		contours=dict(coloring='lines'),
		line=dict(width=3.5), 
		showscale=False
	))

	layout = go.Layout(
		title='Exercise 5 Data',
		xaxis=dict(title='U'),
		yaxis=dict(title='V'))
	plot({"data": data, "layout": layout}, filename=filename)

def get_feature(uv):
	u, v = uv[0], uv[1]
	uv_vec = np.array([1, u, v])
	for i in range(2,7):
		for j in range(0, i+1):
			uv_vec = np.append(uv_vec, u**(i-j) * v**i)
	return uv_vec

x, y = load_data()

if args.shd_plt:
	plot_data(x, y, filename="data.html")

u = x[:,0]
v = x[:,1]
m = x.shape[0]

for i in range(2, 7):
	for j in range(0, i+1):
		u_pow, v_pow = (i-j, j)
		col = np.power(u, u_pow) * np.power(v, v_pow)
		x = np.hstack((x, col.reshape((m,1))))

x = np.hstack((np.ones((m,1)), x))
y = y.reshape((m,1))
n = x.shape[1]

def sigmoid(z):
	#return 1 / (1 + np.exp(-z))
	return expit(z)

def forward_prop(theta, x):
	return sigmoid(np.dot(x, theta.T))

def cost(h, x, y, m, lmbda, theta):
	reg_vec = theta
	reg_vec[0,0] = 0
	log_cost = np.sum(-y*np.log(h) - (1-y)*np.log(1-h))
	reg_cost = lmbda * np.sum(reg_vec**2)/2
	return (log_cost + reg_cost)/m

def grad(h, x, y, m, theta, lmbda):
	reg_vec = theta
	reg_vec[0,0] = 0
	return ((h-y).T.dot(x) + lmbda*reg_vec)/m
	#return (x.T.dot(h-y) + reg_vec)/m

def hessian(h, x, m, n, lmbda, theta):
	hh_1 = np.diag(h[:,0]).dot(np.diag((1-h)[:,0]))
	mat = x.T.dot(hh_1).dot(x)
	reg_mat = np.identity(n)
	reg_mat[0,0] = 0
	return (mat + lmbda*reg_mat)/m

def update(theta, x, y, m, n, lmbda):
	h = forward_prop(theta, x)
	g = grad(h, x, y, m, theta, lmbda).T
	hess = hessian(h, x, m, n, lmbda, theta)
	theta -= np.linalg.solve(hess, g).T
	return g, theta

def theta_for_lambda(lmbda, x, y, m, n, epsilon=1e-10, max_iter=10000):
	#theta = np.zeros((1,n))
	theta = np.zeros((1, n))

	"""num_iter = 0
	last_cost = 1e10
	new_cost = cost(forward_prop(theta, x), x, y, m, lmbda, theta)
	g, theta = update(theta, x, y, m, n, lmbda)
	while last_cost - new_cost > epsilon and num_iter < max_iter:
	#for i in range(15):
		last_cost = new_cost
		new_cost = cost(forward_prop(theta, x), x, y, m, lmbda, theta)
		g, theta = update(theta, x, y, m, n, lmbda)
		num_iter += 1
	return theta[0]"""
	def f(theta, *args):
		theta = theta.reshape((1, n))
		h = forward_prop(theta, x)
		return cost(h, x, y, m, lmbda, theta)
	def fgrad(theta, *args):
		theta = theta.reshape((1, n))
		h = forward_prop(theta, x)
		return grad(h, x, y, m, theta, lmbda).flatten()
	return opt.minimize(f, theta, jac=fgrad, method='CG').x

lambdas = [0, 1, 10]

if __name__ == '__main__':
	thetas = list(map(
		lambda l: theta_for_lambda(l, x, y, m, n), lambdas))
	if args.shd_plt_dec:
		map(
			lambda (i,t): plot_db(
				x, y, t, 
				filename='data_logreg_'+str(i)+'.html'), enumerate(thetas))
	for t in thetas:
		print np.linalg.norm(t)
