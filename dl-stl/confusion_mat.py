import plotly.offline as pltly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import json

from nn_util import *

import argparse
parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--name', dest='name', type=str, default='nn', help='help ya dam self')
args = parser.parse_args()
name = args.name

X, y = load_data(all_data=True, train=False)
m, n = X.shape[0], X.shape[1]
h = n
with open('logs/' + args.name + '/model_sae.json') as f:
	h = json.loads(f.readline())['s'][1]

k = 10 # number of classes
s = []
with open('logs/' + args.name + '/model_nn.json') as f:
	s = json.loads(f.readline())['s']
print s

L = len(s) # number of layers in the network
sizes = [(s[i+1], s[i]+1) for i in range(L-1)]

from sae_util import weights_from_flat
tflat = np.genfromtxt('logs/' + args.name + '/weights_sae.txt')
sae_sizes = [(h, n), (h, 1)]
Ws, bs = weights_from_flat(tflat, sae_sizes)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('logs/'+name+'/weights_nn.txt')
thetas = thetas_from_flat(theta_flat, sizes)

h, pred = predict(X, thetas, Ws, bs, return_h=True)
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
