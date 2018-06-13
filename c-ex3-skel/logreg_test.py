from scipy.io import loadmat
from scipy.special import expit
import numpy as np

import plotly.offline as pltly
import plotly.graph_objs as go

import os
from collections import OrderedDict

from viz import disp_num, show, setup_layout

# calculate the accuracy of the model defined by the given theta matrix
# return both the total accuracy and an array of the accuracy per digit
def calc_acc(theta, X):
	num_correct = 0
	num_correct_num = 0
	acc = np.zeros((10,))

	for i, x in enumerate(X):
		if i % 500 == 0:
			acc[i//500 - 1] = num_correct_num/500.0
			num_correct_num = 0

		correct = i//500
		pred = expit(theta.dot(x))

		if np.argmax(pred) == correct:
			num_correct += 1
			num_correct_num += 1

	acc[-1] = num_correct_num/500.0
	return acc, num_correct/5000.0

# plot the accuracy for different models in a bar chart
def plot_acc(accs, filename='tmp.html'):
	traces = []
	accs = OrderedDict(sorted(accs.items()))
	for name, acc in accs.items():
		tr = go.Bar(x=list(range(10)), y=acc, name=name)
		traces.append(tr)

	layout = go.Layout(
		title='Accuracy per Digit',
		xaxis=dict(title='Digit', dtick=1),
		yaxis=dict(title='Accuracy', range=[0.65, 1]),
		barmode='group')
	pltly.plot({"data": traces, "layout": layout}, filename=filename)	

if __name__ == '__main__':
	mat = loadmat('data/data.mat')
	X = mat['X']
	X = np.hstack((np.ones((X.shape[0],1)), X))

	accs = dict()
	all_accs = dict()
	
	# assume that there are the weights of several different models
	# in a directory names 'weights'.
	# also assume that each filename in this directory follows the format:
	#	weights_<model name>.txt (if there is regularization present in the model,
	#							  then model name will be followed by '_reg')
	files = os.listdir('weights')
	names = list(map(lambda f: f[f.index('_')+1 : f.index('.')], files))
	
	# for each file of weights
	for f, n in zip(files, names):
		weights = np.genfromtxt(os.path.join('weights', f))
		acc_arr, total_acc = calc_acc(weights, X)

		# construct two dictionaries:
		#	accs - maps the name of the model to the array of accuracy
		#	all_accs - maps the name of the model to both the weights and accuracy
		# non-elegant implementation but deal with it
		accs[n] = acc_arr
		all_accs[n] = (weights, acc_arr)
		print 'Total Accuracy for %s: %f.' % (n.upper(), total_acc)

	# construct a dictionary of accuracies for regularized and unregularized models
	reg_acc = {k: v for k, v in accs.items() if 'reg' in k}
	unreg_acc = {k: v for k, v in accs.items() if not k in reg_acc.keys()}
	
	# plot the accuracies
	plot_acc(reg_acc, filename='acc_reg.html')
	plot_acc(unreg_acc, filename='unreg_acc.html')

	setup_layout(4, 2) # create the plot figure with 4 rows and 2 columns
	row, col = -1, 0 # used for grid placement of the current image
	last_n = '' # last_n is the name of the last model, used for updateing row, col
	for n, (weights, acc) in OrderedDict(sorted(all_accs.items())).items():
		print n.upper()

		# get the last model name up to the first '_' (in case it is regularized)
		last_n_start = last_n
		if '_' in last_n:
			last_n_start = last_n[:last_n.index('_')]

		# do the same with the current model
		n_start = n
		if '_' in n:
			n_start = n[:n.index('_')]

		# if it is a new model, increment the row and set the column to 0
		if n_start != last_n_start:
			row += 1
			col = 0
		# otherwise, it is the same model, but regularized so just increment to column
		else:
			col += 1
		last_n = n

		# get the number that the model has the lowest accuracy classifying
		lowest_acc = np.argmin(acc)

		# then get the first image of this number that the model incorrectly classifeis
		# and then plot it along with what was predicted in the title
		idx = lowest_acc * 500
		pred = np.argmax(expit(weights.dot(X[idx])))
		while pred == lowest_acc:
			idx += 1
			pred = np.argmax(expit(weights.dot(X[idx])))

		print 'Correct: %d\tPredicted: %d\n' % (lowest_acc, pred)

		title = '%s; pred %d' % (n, pred)
		disp_num(lowest_acc, X[:,1:], shd_show=False, idx=idx, row=row, col=col, title=title) 
	
	show()	
