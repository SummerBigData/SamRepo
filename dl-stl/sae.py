import argparse

from nn_util import load_data
from sae_util import *

parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('--shd_disp', dest='disp', action='store_true', help='display patches')
parser.add_argument('--check_grad', dest='check', action='store_true', help='check grad')
parser.add_argument('--s', dest='num_samp', type=int, default=1000, help='number to sample')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

X, y= load_data(all_data=True, num_sample=args.num_samp)

m, n = X.shape
s = [n, 196, n]
L = len(s)
sizes = [(s[i+1], s[i]) for i in range(L-1)]

if args.check:
	m = 100
	X = X[:m]
	y = y[:m]

	s = [n, 3, n]
	L = len(s)
	sizes = [(s[i+1], s[i]) for i in range(L-1)]
	
	err = check_grad(X, y, m, n, L, sizes, lmbda=0.01, beta=0.01)
	if err < 1e-4:
		print 'Back prop is probably correct!'
	else:
		print 'Back prop is probably incorrect'
	print err
	exit()

theta_flat = train(X, m, n, L, sizes, lmbda=3e-3, rho=1e-1, beta=3.0, niter=750) 

import os
if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)
np.savetxt('logs/' + args.name + '/weights_sae.txt', theta_flat)

import json
with open('logs/' + args.name + '/model_sae.json', 'w') as f:
	f.write(json.dumps({'s': s}))
