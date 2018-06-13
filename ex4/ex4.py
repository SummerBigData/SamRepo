import numpy as np
from scipy.special import expit

import plotly.offline as pltly
import plotly.graph_objs as go

import argparse
parser = argparse.ArgumentParser(description='Toy linear regression.')
parser.add_argument('--plot', dest='shd_plt', action='store_true', help='Whether or not the data should be plotted')
args = parser.parse_args()

# load and visualize the data
def load_data():
	return np.genfromtxt('data/ex4x.dat'), np.genfromtxt('data/ex4y.dat')

def plot_data(x, y, db=None, filename='tmp.html'):
	admitted = x[np.argwhere(y==1).flatten()]
	not_admitted = x[np.argwhere(y==0).flatten()]
	adm_scatt = go.Scatter(
		x=admitted[:,0], y=admitted[:,1],
		name='Admitted', mode='markers',
		marker=dict(
			size=10, color='rgb(255,0,0)', 
			line=dict(width=1, color='rgb(0,0,0)')))
	nadm_scatt = go.Scatter(
		x=not_admitted[:,0], y=not_admitted[:,1],
		name='Not Admitted', mode='markers',
		marker=dict(
			size=10, color='rgb(0,0,255)',
			line=dict(width=1, color='rgb(0,0,0)')))
	data = [adm_scatt, nadm_scatt]

	if not db is None:
		dec_bound = go.Scatter(
			x=db[0], y=db[1],
			name='Decision Boundary', mode='lines')
		data.append(dec_bound)
	
	layout = go.Layout(
		title='College Admissions',
		xaxis=dict(title='Test1 Score'),
		yaxis=dict(title='Test2 Score'))
	pltly.plot({
		"data": data,
		"layout": layout}, filename=filename)

def plot_cost(J, filename='cost.html'):
	scatter = go.Scatter(
		x=list(range(len(J))),
		y=J, mode='lines')
	layout = go.Layout(
		title='Cost over Time',
		xaxis=dict(title='Iteration'),
		yaxis=dict(title='Cost'))
	pltly.plot({
		"data": [scatter],
		"layout": layout}, filename=filename)

x, y = load_data()
if args.shd_plt:
	plot_data(x, y, filename="data.html")

#sigma = np.std(x, axis=0)
#mu = np.std(x, axis=0)
#for i in range(len(sigma)):
#	x[:,i] = (x[:,i] - mu[i]) / sigma[i]

m, n = x.shape[0], x.shape[1]
x = np.hstack((np.ones((m,1)), x))
y = np.reshape(y, (m,1))
theta = np.zeros((1, n+1), dtype=float)

def sigmoid(z):
	#return 1 / (1 + np.exp(-z))
	return expit(z)

def forward_prop(theta, x):
	return sigmoid(np.dot(x, theta.T))

def cost(h, x, y, m):
	return np.sum(
		y*np.log(h) + (1-y)*np.log(1-h))/(-m)

def grad(h, x, y, m):
	return np.dot((h - y).T, x)/m

def hessian(h, x, m):
	hh_1 = np.diag(h[:,0]).dot(np.diag((1 - h)[:,0]))
	return x.T.dot(hh_1).dot(x) / m
	#hh_1 = (h[:,0]).T.dot(1-h[:,0])
	#hess = np.zeros((3,3))
	#for row in x:
	#	hess += np.outer(row, row)
	#return hh_1 * hess / m

def update(theta, x, y, m):
	h = forward_prop(theta, x)
	g = grad(h, x, y, m).T
	hess = hessian(h, x, m)
	#theta -= np.dot(np.linalg.inv(hess), g).T
	theta -= np.linalg.solve(hess, g).T
	return g, theta

epsilon = 1e-4
num_iter = 0
max_iter = 10000

J = np.array([cost(forward_prop(theta, x), x, y, m)])
g, theta = update(theta, x, y, m)
while abs(np.sum(g)) > epsilon and num_iter < max_iter:
	J = np.append(J, np.array([cost(forward_prop(theta, x), x, y, m)]))
	g, theta = update(theta, x, y, m)
	num_iter += 1

theta = theta[0]
#for i in range(1, len(sigma)):
#	theta[i] = theta[i] * sigma[i] + mu[i]
print 'After %d iterations, theta0 = %f, theta1 = %f, theta2 = %f' % (num_iter, theta[0], theta[1], theta[2])

x = x[:,1:]
y = y[:,0]

left_pt_x = min(x[:,0])
right_pt_x = max(x[:,0])
left_pt_y = -(theta[0] + left_pt_x*theta[1])/theta[2]
right_pt_y = -(theta[0] + right_pt_x*theta[1])/theta[2]
plot_data(x, y, db=[[left_pt_x, right_pt_x], [left_pt_y, right_pt_y]], filename='data_logreg.html')
plot_cost(J, filename='cost.html')

test_x = np.array([1.0, 20.0, 80.0]).reshape((3,1))
prob_adm = sigmoid(np.dot(theta, test_x))
print 'Given 20 on Exam 1 and 80 on Exam 2, there is a %f percent chance that this student will not be admitted.' % (1-prob_adm,)
