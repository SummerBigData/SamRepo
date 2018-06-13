import numpy as np

from nn_util import load_data
from sae_util import *
from display import display_encoding, display_decoding

import argparse
parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

X, y = load_data(all_data=True, train=False)

s = []
with open('logs/' + args.name + '/model_sae.json') as f:
    import json
    s = json.loads(f.readline())['s']
L = len(s)
sizes = [(s[i+1], s[i]) for i in range(L-1)]

tmp_sizes = []
for size in sizes:
    tmp_sizes.append(size)
    tmp_sizes.append((size[0], 1))
sizes = tmp_sizes[:]

tflat = np.genfromtxt('logs/' + args.name + '/weights_sae.txt')
Ws, bs = weights_from_flat(tflat, sizes)
a_arr, _ = forward_prop(X, Ws, bs)

#display_encoding(Ws[0])
for i in np.random.choice(list(range(len(X))), 5, replace=False):
    display_decoding(X[i], a_arr[-1][i])
