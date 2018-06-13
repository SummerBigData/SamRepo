import plotly.offline as pltly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import json

from nn_util import *

import argparse 
parser = argparse.ArgumentParser(description='make a confusion matrix')
parser.add_argument('--all_data', dest='all_data', help='Whether or not to use the whole dataset', action='store_true')
parser.add_argument('--name', dest='name', help='The name of the model', type=str, default='nn')

args = parser.parse_args()
if args.all_data:
	X, y = load_data(all_data=True, train=False)
else:
	_, _, X, y = load_data(split=True)

m, n = X.shape[0], X.shape[1]
k = 10
s = []
with open('logs/'+args.name+'/model.json') as f:
	s = json.loads(f.readline())['s']
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('logs/'+args.name+'/weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

pred = predict(X, thetas)
actual = np.argmax(y, axis=1)

z = []
for i in range(k):
	pred_vec = pred[np.argwhere(actual == i).flatten()]
	confusion_vec = []
	for j in range(k):
		confusion_vec.append(np.sum(pred_vec == j))
	z.append(confusion_vec)

fig = ff.create_annotated_heatmap(z)
fig.layout['title'] = 'Confusion matrix'
fig.layout['xaxis'] = dict(title='Predicted', dtick=1)
fig.layout['yaxis'] = dict(title='Actual', dtick=1)
pltly.plot(fig, filename='confusion_matrix.html')
#pltly.plot({"data": data, "layout": layout}, filename="confusion_matrix.html")
