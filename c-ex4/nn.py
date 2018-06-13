import numpy as np
import argparse
import json
import atexit
import os

from nn_util import *
from prepro import *

# First and foremost, I would like to apologize for the horrendous mess you are about to witness
# Due to adding functionality and a hint of laziness, many functions take weird arguments and do many different things
# Sorry.

# add hella arguments
parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--size_h', dest='size_h', help='Size of the hidden layer', type=int, default=25)
parser.add_argument('--num_h', dest='num_h', help='Number of hidden layers in the network', type=int, default=1)
parser.add_argument('--lambda', dest='lmbda', help='Regularization hyperparameter', type=int, default=1)
parser.add_argument('--check_grad', dest='check_grad', action='store_true', help='Whether or not to numerical check the gradient from backpropagation (very expensive)')
parser.add_argument('--all_data', dest='use_all_data', action='store_true', help='Whether or not to use to full dataset')
parser.add_argument('--max_iter', dest='max_iter', help='Max number of iteration for optimization', type=int, default=200)
parser.add_argument('--shd_split', dest='split', help='Whether or not to split the dataset read into training and test sets', action='store_true')
parser.add_argument('--name', dest='name', help='The name of the program (for logs and plots)', type=str, default='nn')
parser.add_argument('--num_samp', dest='num_sample', help='The number of the samples to use from the actual dataset', type=int, default=500)
parser.add_argument('--shd_disp', dest='display', action='store_true', help='Whether or not to display some test digits to test read functionality')
parser.add_argument('--aug_mode', dest='aug_mode', help='0 - no augmentation, 1 - duplication, \
                                                        2 - just rotation, 3 - just scaling, \
                                                        4- just translation, 5 - both rotation and scaling, \
                                                        6 - both rotation and translation, 7 - both scaling and translation, \
                                                        8 - all three', type=int, default=0)
parser.add_argument('--dwn_samp', dest='down_samp', help='What digit to downsample, if any', type=int, default=-1)
args = parser.parse_args()

print 'Training: ' + args.name.upper()

if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)

""" Loading the data """
# initialize a test set to None. If the toy dataset is being used then take the last 10th of the data as a
# training set and pass it to the minimize function for accuracy tracking.

# if we are using the full dataset, then we have separate files for the test set and we can load those when needed
# in the aforementioned function
X_test, y_test = None, None
if args.split:
	X, y, X_test, y_test = load_data(all_data=args.use_all_data, split=True)
else:
	X, y = load_data(all_data=args.use_all_data, num_sample=args.num_sample)


ds_dig = args.down_samp
aug_mode = args.aug_mode
if ds_dig >= 0 and aug_mode > 0:
    X, y = augment(X, y, ds_dig, aug_mode)

if args.display:
    from display import *
    display_tests(X, y)
    exit()

print X.shape, y.shape


""" Setting appropriate constants """
m, n = X.shape[0], X.shape[1]
k = 10 # number of classes
s = [n]
s.extend([args.size_h] * args.num_h) 
for i, l in enumerate(s[1:]):
	s[i+1] = max(10, l//(i+1))
s.append(k)
print s
L = len(s) # number of layers in the network

""" model saving """
# save the model architecture to a json file
with open('logs/'+args.name+'/model.json', 'w') as f:
	f.write(json.dumps({'s': s}))


""" Gradient checking """
# if we pass the --check_grad argument, don't train anything. Create a smaller network (because gradient checking
# is slow af) and see if the backpropagation matches the numerical gradient to within 10^-5
if args.check_grad:
	m = 100
	X = X[:m]
	y = y[:m]

	s = [n, 3, k]
	sizes = [(s[i+1], s[i]+1) for i in range(L-1)]

	err = check_grad(X, y, m, n, k, L, sizes, lmbda=1)
	print err
	if err < 1e-5:
		print 'Back prop is probably correct!'
	else:
		print 'Back prop is probably incorrect :('
	exit()


# if we are about to crash or PBS is kicking us off, then write out the statistics that are being tracked (cost and accuracy)
# also save the weights if we got close to finishing but got kicked off of PBS
atexit.register(write_cost_and_acc)
atexit.register(write_last_theta)

""" Training """
sizes = [(s[i+1], s[i]+1) for i in range(L-1)]

# sorry for all of the arguments, I know it's ugly
theta_flat = train(X, y, m, n, k, L, sizes, lmbda=args.lmbda, all_data=args.use_all_data, max_iter=args.max_iter, test_set=(X_test, y_test), name=args.name)
np.savetxt('logs/'+args.name+'/weights.txt', theta_flat)
