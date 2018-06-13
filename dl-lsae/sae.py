import scipy.io as sio

import argparse

from util import *

parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('-s', dest='s', type=int, default=1000)
parser.add_argument('--shd_disp', dest='disp', action='store_true', help='display patches')
parser.add_argument('--check_grad', dest='check', action='store_true', help='check grad')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

mat = sio.loadmat('data/patches.mat') 
X = mat['patches'].T
X = X[:args.s]
mu = np.mean(X, axis=0)
np.savetxt('mu.txt', mu)

X, zcaw = zca(X)
np.savetxt('zcaw.txt', zcaw)

m, n = X.shape
s = [n, 400, n]
L = len(s)
sizes = [(s[i+1], s[i]) for i in range(L-1)]

if args.disp:
	from display import display_sample
	display_sample(X)
	exit()

if args.check:
	m = 100
	X = X[:m]

	s = [n, 3, n]
	L = len(s)
	sizes = [(s[i+1], s[i]) for i in range(L-1)]
	
	err = check_grad(X, X, m, n, L, sizes, lmbda=0.01, beta=0.01)
	print err
	if err < 1e-4:
		print 'Back prop is probably correct!'
	else:
		print 'Back prop is probably incorrect'
	exit()

import os
if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)
theta_flat = train(X, m, n, L, sizes, lmbda=3e-3, rho=0.035, beta=5.0, niter=750, name=args.name) 
np.savetxt('logs/' + args.name + '/weights.txt', theta_flat)

import json
with open('logs/' + args.name + '/model.json', 'w') as f:
	f.write(json.dumps({'s': s}))
