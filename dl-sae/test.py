import scipy.io as sio
import numpy as np

from util import *
from display import display_encoding, display_decoding

import argparse
parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

mat = sio.loadmat('data/images.mat')
imgs = mat['IMAGES']
imgs = imgs.reshape(imgs.shape[0]**2, imgs.shape[2]).T

X = generate_sample(imgs)
y = X.copy()

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

display_encoding(Ws[0])
#display_decoding(X[0], a_arr[-1][0])