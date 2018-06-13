import matplotlib.pyplot as plt
import numpy as np
from random import randint
from collections import Counter
import argparse
import json

from nn_util import *

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
actual = np.argmax(y, axis=1)

max_fn = 0
fns = []

for i in range(k):
    idxs = np.argwhere(actual == i).flatten()

    pred_vals = pred[idxs].flatten()
    most_comm_wrong = Counter(pred_vals.tolist()).most_common(2)[1]
    print i, most_comm_wrong

    lowest_acc = most_comm_wrong[0]
    wrongs = X[idxs][np.argwhere(pred_vals == lowest_acc).flatten()]
    fns.append(wrongs)
    print h[idxs][np.argwhere(pred_vals == lowest_acc), lowest_acc].flatten()
    print h[idxs][np.argwhere(pred_vals == lowest_acc), i].flatten()
    print ''

    if len(wrongs) > max_fn:
        max_fn = len(wrongs)

"""
dim = int(np.sqrt(len(X[0])))
rows, cols = 10, max_fn
_, axs = plt.subplots(rows, cols, figsize=(rows, cols))

row = 0
for fn_vec in fns:
    col = 0
    i  = 0
    for x in fn_vec:
        x = x.reshape((dim, dim))
        x *= 255
        x = 1-x

        ax = axs[row, col]
        ax.imshow(x, cmap='gray')
        ax.axis('off')

        col += 1
        i += 1

    while i < max_fn:
        axs[row, i].axis('off')
        i += 1

    row += 1

plt.tight_layout()
plt.show()
"""