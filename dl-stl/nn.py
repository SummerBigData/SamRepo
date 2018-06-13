import numpy as np
import argparse
import json
import atexit
import os

from nn_util import *

# First and foremost, I would like to apologize for the horrendous mess you are about to witness
# Due to adding functionality and a hint of laziness, many functions take weird arguments and do many different things
# Sorry.

# add hella arguments
parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--size_h', dest='size_h', help='Size of the hidden layer', type=int, default=25)
parser.add_argument('--num_h', dest='num_h', help='Number of hidden layers in the network', type=int, default=1)
parser.add_argument('--lambda', dest='lmbda', help='Regularization hyperparameter', type=int, default=1)
parser.add_argument('--check_grad', dest='check_grad', action='store_true', help='Whether or not to numerical check the gradient from backpropagation (very expensive)')
parser.add_argument('--name', dest='name', help='The name of the program (for logs and plots)', type=str, default='nn')
parser.add_argument('-s', dest='num_sample', help='The number of the samples to use from the actual dataset', type=int, default=500)
parser.add_argument('--max_iter', dest='max_iter', help='Maximum number of iterations for CG', type=int, default=500)
args = parser.parse_args()

print 'Training: ' + args.name.upper()

if not os.path.isdir('logs/'+args.name):
	print 'No existing logs directory for ', args.name
	exit()

""" Loading the data """
# initialize a test set to None. If the toy dataset is being used then take the last 10th of the data as a
# training set and pass it to the minimize function for accuracy tracking.

X_test, y_test = None, None
X, y = load_data(all_data=True, train=True, num_sample=args.num_sample)
print X.shape, y.shape

""" Setting appropriate constants """
m, n = X.shape[0], X.shape[1]
h = n
with open('logs/' + args.name + '/model_sae.json') as f:
	h = json.loads(f.readline())['s'][1]

k = 10 # number of classes
s = [h]
s.extend([args.size_h] * args.num_h) 
s.append(k)
print s

L = len(s) # number of layers in the network
sizes = [(s[i+1], s[i]+1) for i in range(L-1)]

from sae_util import weights_from_flat
tflat = np.genfromtxt('logs/' + args.name + '/weights_sae.txt')
sae_sizes = [(h, n), (h, 1)]
Ws, bs = weights_from_flat(tflat, sae_sizes)

""" model saving """
# save the model architecture to a json file
with open('logs/'+args.name+'/model_nn.json', 'w') as f:
	f.write(json.dumps({'s': s}))


""" Gradient checking """
# if we pass the --check_grad argument, don't train anything. Create a smaller network (because gradient checking
# is slow af) and see if the backpropagation matches the numerical gradient to within 10^-5
if args.check_grad:
    m = 100
    X = X[:m]
    y = y[:m]

    if len(s) > 2:
        s = s[:3]
        s[1] = 3
    sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
    print s

    err = check_grad(X, y, Ws, bs, m, n, k, L, sizes, lmbda=args.lmbda)
    print err
    if err < 1e-4:
        print 'Back prop is probably correct!'
    else:
        print 'Back prop is probably incorrect :('
    exit()


""" Training """
# sorry for all of the arguments, I know it's ugly
theta_flat = train(X, y, Ws, bs, m, n, k, L, sizes, lmbda=args.lmbda, max_iter=args.max_iter, test_set=(X_test, y_test), name=args.name)
np.savetxt('logs/'+args.name+'/weights_nn.txt', theta_flat)
