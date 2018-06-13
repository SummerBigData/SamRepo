import scipy.io as sio
import numpy as np

from util import *
from display import display_encoding, display_decoding

import argparse
parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

mat = sio.loadmat('data/patches.mat') 
X = mat['patches'].T
X = X[:1000]
X, zca_white = zca(X)
m, n = X.shape

s = []
with open('logs/' + args.name + '/model.json') as f:
    import json
    s = json.loads(f.readline())['s']
L = len(s)
sizes = [(s[i+1], s[i]) for i in range(L-1)]

tmp_sizes = []
for size in sizes:
    tmp_sizes.append(size)
    tmp_sizes.append((size[0], 1))
sizes = tmp_sizes[:]

tflat = np.genfromtxt('logs/' + args.name + '/weights.txt')
Ws, bs = thetas_from_flat(tflat, sizes)
a_arr, _ = forward_prop(X, Ws, bs)

#display_encoding(Ws[0])
from display import display_sample
display_encoding(Ws[0].dot(zca_white))
#display_decoding(X[0], a_arr[-1][0])