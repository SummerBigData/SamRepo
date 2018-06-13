from scipy.io import loadmat
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize as opt
import numpy as np

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

# generate an array of random matrices according to the given sizes
def random_thetas(sizes):
    def eps(size):
        return np.sqrt(6.0/(size[0]+size[1]+1.0))
    return [np.random.uniform(-eps(s), eps(s), s) for s in sizes]

# calculate the cost of the model
def cost(h, y, m, k, lmbda, theta):
    cost = -np.mean(np.sum(y*np.log(h), axis=1), axis=0)
    if lmbda > 0:
        weight_sum = np.sum(theta[:,1:].flatten()**2)
        cost += weight_sum * lmbda / (2.0)
    return cost

""" FORWARD PROP """
# propagate the dataset forward through the network given a set of parameters
def forward_prop(X, theta):
    a = X
    a_arr = []

    a = np.hstack((np.ones((len(a), 1)), a))
    a_arr.append(a)
    z = a.dot(theta.T)
    a = np.exp(z)
    denom = np.sum(a, axis=1)
    a = a / denom.reshape(len(a), 1)

    a_arr.append(a)
    return a_arr

""" BACKPROP """
# using backpropagation, calculate the gradient with respect to all weights of the model
def back_prop(a_arr, y, theta, lmbda):
    m = float(len(y))

    for i in range(m):
        lil_del[-1] = -(y[i] - a_arr[-1][i])
        lil_del[-1] = lil_del[-1].reshape((k, 1))

        for l in range(2, L+1):
            last_delta = lil_del[-l+1]
            if l > 2: # if it is a hidden layer then ignore the bias error
                last_delta = last_delta[1:,:]

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
        big_del[i] = bd/float(m)
        if lmbda > 0:
            reg_mat = thetas[i][:]
            reg_mat[:,0] = 0
            big_del[i] += lmbda*reg_mat/float(m)
    
    return big_del

def predict(X, thetas, Ws, bs, return_h=False):
    X = sae_util.forward_prop(X, Ws, bs)[0][1]
    h = forward_prop(X, thetas)[-1]
    pred = np.argmax(h, axis=1)
    if return_h:
        return (h, pred)
    return pred

def thetas_from_flat(theta_flat, sizes):
    thetas = []
    idx = 0
    for size in sizes:
        n = size[0]*size[1]
        thetas.append(
            np.array(theta_flat[idx:idx+n]).reshape(size))
        idx += n
    return thetas

def check_grad(X, y, Ws, bs, m, n, k, L, sizes, lmbda=0):
    print 'Calculating numerical gradient'
    thetas = random_thetas(sizes)
    theta_flat = np.array([], dtype=float)
    for theta in thetas:
        theta_flat = np.append(theta_flat, theta.flatten())

    X = sae_util.forward_prop(X, Ws, bs)[0][1]
    print X.shape

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
    np.savetxt('logs/'+prog_name+'/weights_nn.txt', last_tflat)

# train a network using convergent gradient minimization
def train(X, y, Ws, bs, m, n, k, L, sizes, lmbda=0, max_iter=400, test_set=None, name='nn'):
    global global_step, Js, accs, prog_name, last_tflat
    
    prog_name = name
    X_test, y_test = test_set
    if X_test is None or y_test is None:
        X_test, y_test = load_test(all_data=True)

    X = sae_util.forward_prop(X, Ws, bs)[0][1]
    X_test = sae_util.forward_prop(X_test, Ws, bs)[0][1]

    thetas = random_thetas(sizes)
    theta_flat = np.array([], dtype=float)

    for theta in thetas:
        theta_flat = np.append(theta_flat, theta.flatten())

    def f(thetas, *args):
        global last_tflat, Js, accs, global_step 
        last_tflat = np.copy(thetas) # store the last theta for if we get booted

        thetas = thetas_from_flat(thetas, sizes)
        h = forward_prop(X, thetas)[-1]
        J = cost(h, y, m, k, lmbda=lmbda, thetas=thetas)

        if global_step % 20 == 0:
            Js.append(J)
            accs.append(calc_acc(X_test, y_test, thetas))
            print Js[-1], accs[-1]
            write_cost_and_acc()
            write_last_theta()
            
        return J
    
    def fgrad(thetas, *args):
        global last_tflat, Js, accs
        last_tflat = np.copy(thetas) # store the last theta vector in case we have to exit early

        thetas = thetas_from_flat(thetas, sizes)
        a_arr = forward_prop(X, thetas)	
        dels = back_prop(a_arr, y, m, L, thetas, lmbda=lmbda)

        g = np.array([], dtype=float)
        for d in dels:
            g = np.append(g, d.flatten())
        return g

    return opt.minimize(
        f, theta_flat, jac=fgrad, 
        method='CG', tol=1e-5, 
        options={'disp': True, 'maxiter': max_iter}).x

