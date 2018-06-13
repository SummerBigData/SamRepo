import numpy as np
import argparse
import json
import os

from cnn_util import *

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--name', dest='name', help='The name of the program (for logs and plots)', type=str, default='test')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=4)
parser.add_argument('--test', dest='test', action='store_true')
args = parser.parse_args()

if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)

X, _ = load_data(train=(not args.test), shuffle=False)
img_len = X.shape[1] * X.shape[2] * X.shape[3]
zcaw = np.genfromtxt('zca_sae.txt')
mu = np.genfromtxt('mu_sae.txt')
mu = mu.reshape(len(mu), 1)

m, n = X.shape[0], img_len
h = n
with open('model_lsae.json') as f:
	h = json.loads(f.readline())['s'][1]

tflat = np.genfromtxt('weights_lsae.txt')
wlen = X.shape[1] * X.shape[3]
W = tflat[:wlen*h].reshape(h, wlen)
b = tflat[wlen*h:wlen*h+h].reshape(h, 1)

X_conv = convolve_and_pool(X, W, b, zcaw, mu, num_proc=args.num_proc).reshape(m, -1)
name = '/X_conv_train.txt'
if args.test:
    name = '/X_conv_test.txt'
np.savetxt('logs/'+args.name+name, X_conv)