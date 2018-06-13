import scipy.io as sio

import argparse

from util import *

parser = argparse.ArgumentParser(description='Sparse Autoencoder')
parser.add_argument('--shd_disp', dest='disp', action='store_true', help='display patches')
parser.add_argument('--check_grad', dest='check', action='store_true', help='check grad')
parser.add_argument('--name', dest='name', type=str, help='name')
args = parser.parse_args()

mat = sio.loadmat('data/images.mat')
imgs = mat['IMAGES']
imgs = imgs.reshape(imgs.shape[0]**2, imgs.shape[2]).T

X = generate_sample(imgs)
m, n = X.shape
s = [n, 25, n]
L = len(s)
sizes = [(s[i+1], s[i]) for i in range(L-1)]

if args.disp:
	from display import display_sample
	display_sample(X)
	exit()

if args.check:
	m = 100
	X = X[:m]
	y = y[:m]	

	s = [n, 3, n]
	L = len(s)
	sizes = [(s[i+1], s[i]) for i in range(L-1)]
	
	err = check_grad(X, y, m, n, L, sizes, lmbda=0.01, beta=0.01)
	print err
	if err < 1e-4:
		print 'Back prop is probably correct!'
	else:
		print 'Back prop is probably incorrect'
	exit()

theta_flat = train(X, m, n, L, sizes, lmbda=1e-4, rho=1e-2, beta=3.0, niter=750) 

import os
if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)
np.savetxt('logs/' + args.name + '/weights.txt', theta_flat)

import json
with open('logs/' + args.name + '/model.json', 'w') as f:
	f.write(json.dumps({'s': s}))
