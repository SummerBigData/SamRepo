from scipy.io import loadmat
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize as opt
import numpy as np

from read_mnist import *
#from prepro import crop
import os
from functools import reduce

""" Global variables """ # I know they're gross
Js, accs = [], [] # lists for tracking the cost and accuracy over time
prog_name = 'nn' # the name of the program for avoiding collisions in log files
last_tflat = np.zeros((1,1)) # the last theta vector if we need to save it early
global_step = 0 # tracks how many times the back_propagation algorithm is called


""" Data loading """
def shuffle(X, y):
    X = np.hstack((X, y.reshape((len(y), 1))))
    np.random.shuffle(X)
    return X[:,:-1], X[:,-1]

def onehot(y):
    onehot_encoder = OneHotEncoder(sparse=False)
    return onehot_encoder.fit_transform(y.reshape((len(y), 1))) # onehot encode y

def shuffle_oh(X, y):
    X, y = shuffle(X, y)
    return X, onehot(y)

# load the data from the .mat file and do some preprocessing
def load_data(all_data=False, train=True, split=False, num_sample=500):
    # if we are using the full dataset, we can load it from the -ubyte.gz files
    # if we are using the toy dataset, we can load it from the .mat files

    # using the full dataset, we have a train and a test set, to the train argument tells us
    # which dataset to read

    # using the toy dataset, there is no test set, so split should be set to True so that
    # we can say that the last 10th of the train set is actually the test set

    # if you are using the full dataset, don't set split to True, that would just be pointless

    if all_data:
        filenames = os.listdir('./data')
        filenames = [f for f in filenames if f.endswith('.gz')]
        filenames = [os.path.join('./data', f) for f in filenames]

        if train:
            filenames = [f for f in filenames if 'train' in f]
        else:
            filenames = [f for f in filenames if not 'train' in f]

        if not train:
            num_sample = None

        if 'images' in filenames[0]:
            #X, y = read_images(filenames[0], n=num_sample), read_labels(filenames[1], n=num_sample)
            X, y = read_idx(filenames[0], n=num_sample), read_idx(filenames[1], n=num_sample)
        else:
            #X, y = read_images(filenames[1], n=num_sample), read_labels(filenames[0], n=num_sample)
            X, y = read_idx(filenames[1], n=num_sample), read_idx(filenames[0], n=num_sample)
        #X = crop(X)
        X = X / 255.0
        X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))


    else:
        mat = loadmat('data/data.mat')
        X, y = mat['X'], mat['y']
        prepro = lambda i: 0 if i == 10 else i
        v_prepro = np.vectorize(prepro)
        y = v_prepro(y.flatten())



    if not all_data:
        if split:
            test_len = 500//10
            idxs = reduce(
                lambda x, y: x+y,
                [[500*i-j for j in reversed(range(1,test_len+1))] for i in range(1,11)])

            X_test, y_test = X[idxs], y[idxs]
            idxs = [i for i in range(5000) if not i in idxs]

            train, test = shuffle_oh(X[idxs], y[idxs]), shuffle_oh(X_test, y_test)
            return train[0], train[1], test[0], test[1]
        
        return shuffle_oh(X, y)

    return X, onehot(y)



# generate an array of random matrices according to the given sizes
def random_thetas(sizes, epsilon=0.12):
    return [np.random.randn(size[0], size[1])*2*epsilon-epsilon for size in sizes]



# calculate the cost of the model
def cost(h, y, m, k, lmbda=0, thetas=[]):
    cost = np.sum(
        -np.sum(
            y * np.log(h) + (1-y) * np.log(1 - h),
            axis=1),
        axis=0) / m
    if lmbda > 0:
        weight_sum = sum(list(map(
            lambda t: np.sum(t[:,1:].flatten()**2),
            thetas)))
        cost += weight_sum * lmbda / (2.0)
    return cost



""" FORWARD PROP """
# propagate the dataset forward through the network given a set of parameters
def forward_prop(X, thetas):
    a = X
    a_arr = []

    for theta in thetas:
        a = np.hstack((np.ones((len(a), 1)), a))
        a_arr.append(a)
        z = a.dot(theta.T)
        a = expit(z)

    a_arr.append(a)
    return a_arr



""" BACKPROP """
# using backpropagation, calculate the gradient with respect to all weights of the model
def back_prop(a_arr, y, m, L, thetas, lmbda=0):
    global global_step
    global_step += 1
    if global_step % 20 == 0:
        print 'Global Step: %d' % (global_step)

    big_del, lil_del = [], []
    k = len(y[0])

    # create an array for the delta error of each later than where each element
    # is the shape of (# of neurons, 1) per layer
    for a in a_arr:
        lil_del.append(np.zeros((a[0].shape[0], 1)))


    # create an array for the cumulative theta matrix for each layer (the size of theta)
    for t in thetas:
        big_del.append(np.zeros(t.shape))

    for i in range(m):
        # initialize the output error to be activation - y
        lil_del[-1] = (a_arr[-1][i] - y[i])
        lil_del[-1] = lil_del[-1].reshape((k, 1))

        # for each subsequent layer, use the chain rule to calculate errors
        for l in range(2, L+1):
            # get the error of the layer above
            last_delta = lil_del[-l+1]
            if l > 2: # if it is a hidden layer then ignore the bias error
                last_delta = last_delta[1:,:]

            # compute the error of the current layer
            curr_delta = thetas[-l+1].T.dot(last_delta)

            a = a_arr[-l][i] # get this layer's activation
            s = len(a) # size of the current layer
            a = a.reshape((s, 1))

            big_del[-l+1] += last_delta.dot(a.T) # increment big delta for the layer accordingly

            gprime = np.multiply(a, 1-a)
            gprime = gprime.reshape((s, 1))

            curr_delta = np.multiply(curr_delta, gprime) # multiply the error by it
            lil_del[-l][:] = curr_delta[:]

    for i, bd in enumerate(big_del):
        big_del[i] = bd/m
        if lmbda > 0:
            reg_mat = thetas[i][:]
            reg_mat[:,0] = 0
            big_del[i] += lmbda*reg_mat

    return big_del


def predict(X, thetas, return_h=False):
    h = forward_prop(X, thetas)[-1]
    pred = np.argmax(h, axis=1)
    if return_h:
        return (h, pred)
    return pred


# return an array of matrices from the flattened parameter vectors according to the passed sizes
def thetas_from_flat(theta_flat, sizes):
    thetas = []
    idx = 0
    for size in sizes:
        n = size[0]*size[1]
        thetas.append(
            np.array(theta_flat[idx:idx+n]).reshape(size))
        idx += n
    return thetas


def check_grad(X, y, m, n, k, L, sizes, lmbda=0):
    print 'Calculating numerical gradient'
    thetas = random_thetas(sizes)
    theta_flat = np.array([], dtype=float)
    for theta in thetas:
        theta_flat = np.append(theta_flat, theta.flatten())

    def f(thetas, *args):
        thetas = thetas_from_flat(thetas, sizes)
        h = forward_prop(X, thetas)[-1]
        return cost(h, y, m, k, lmbda=lmbda, thetas=thetas)
    
    def fgrad(thetas, *args):
        thetas = thetas_from_flat(thetas, sizes)
        a_arr = forward_prop(X, thetas)	
        dels = back_prop(a_arr, y, m, L, thetas, lmbda=lmbda)
        g = np.array([], dtype=float)
        for d in dels:
            g = np.append(g, d.flatten())
        return g

    return opt.check_grad(f, fgrad, theta_flat)

def load_test(all_data=False):
    return load_data(all_data=all_data, train=False)

def calc_acc(X_test, y_test, thetas):
    h = forward_prop(X_test, thetas)[-1]
    pred = np.argmax(h, axis=1)
    actual = np.argmax(y_test, axis=1)
    return np.sum(pred == actual) / float(len(y_test))

def write_cost_and_acc():
	global Js, accs
	with open('logs/'+prog_name+'/cost_and_acc.csv', 'w') as f:
		f.write('Cost, Accuracy\n')
		for j, acc in zip(Js, accs):
			f.write('%f, %f\n' % (j, acc))

def write_last_theta():
	global last_tflat, prog_name
	np.savetxt('logs/'+prog_name+'/weights.txt', last_tflat)


# train a network using convergent gradient minimization
def train(X, y, m, n, k, L, sizes, lmbda=0, all_data=False, max_iter=200, test_set=None, name='nn'):
	global global_step, Js, accs, prog_name, last_tflat
	
	prog_name = name
	X_test, y_test = test_set
	if X_test is None or y_test is None:
		X_test, y_test = load_test(all_data=all_data)


	""" Parameter initialization """
	# create a random matrix of thetas for each layer
	thetas = random_thetas(sizes)
	theta_flat = np.array([], dtype=float)

	# flatten that list to make it minimizable using scipy
	for theta in thetas:
		theta_flat = np.append(theta_flat, theta.flatten())


	# In order to avoid passing arguments from scipy.optimize.minimize to cost and grad like a barbarian,
	# I created two nested functions that simply take the flattened theta vector and use the parameters
	# passed to the greater train function when needed

	""" Nested cost """
	def f(thetas, *args):
		global last_tflat, Js, accs, global_step 
		last_tflat = np.copy(thetas) # store the last theta for if we get booted

		""" Do the actual calculation """
		thetas = thetas_from_flat(thetas, sizes)
		h = forward_prop(X, thetas)[-1]
		J = cost(h, y, m, k, lmbda=lmbda, thetas=thetas)

		# Every 20 iterations, store the cost and the accuracy just in case we get booted
		# and to make sure that we are still improving as time goes on
		if global_step % 20 == 0:
			Js.append(J)
			accs.append(calc_acc(X_test, y_test, thetas))
			print Js[-1], accs[-1]
			write_cost_and_acc()
			write_last_theta()
			
		return J
	
	""" Nested gradient of the cost """
	def fgrad(thetas, *args):
		global last_tflat, Js, accs
		last_tflat = np.copy(thetas) # store the last theta vector in case we have to exit early

		""" Do the actual calculation """
		thetas = thetas_from_flat(thetas, sizes)
		a_arr = forward_prop(X, thetas)	
		dels = back_prop(a_arr, y, m, L, thetas, lmbda=lmbda)

		# once again, flatten the list of gradient matrices for scipy
		g = np.array([], dtype=float)
		for d in dels:
			g = np.append(g, d.flatten())
		return g

	return opt.minimize(
		f, theta_flat, jac=fgrad, 
		method='CG', tol=1e-3, 
	#	method='CG',
		options={'disp': True, 'maxiter': max_iter}).x

