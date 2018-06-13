from scipy.io import loadmat
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize as opt
import numpy as np

import os
from functools import reduce

from nn_util import random_thetas, thetas_from_flat

""" Global variables """ # I know they're gross
Js, accs = [], [] # lists for tracking the cost and accuracy over time
prog_name = 'nn' # the name of the program for avoiding collisions in log files
last_tflat = np.zeros((1,1)) # the last theta vector if we need to save it early
global_step = 0 # tracks how many times the back_propagation algorithm is called

""" Utility for mapping and returning a list like it should do by default """
def lmap(f, items):
    return list(map(f, items))

""" Shorthand for wrapping a list in a Numpy array since it's common is cluttered """
def a(items):
    return np.array(items)

""" Shorthand for enumerate since I use it alot """
def e(items):
    return enumerate(items)

def random_theta(size):
    def eps(size):
        return np.sqrt(6.0/(size[0]+size[1]+1.0))
    return np.random.uniform(-eps(size), eps(size), size)

def cost(h, y, l, thetas):
    cost = -np.mean(np.sum(y*np.log(h), axis=1), axis=0)
    #weight_sum = sum(lmap(lambda T: np.sum(T[:,1:].flatten()**2), thetas))
    #return cost + (l / 2.0) * weight_sum
    cost += l / 2.0 * np.sum(thetas[-1][:,1:].flatten()**2)
    # cost += l / 2.0 * np.sum(thetas[-2][:,1:].flatten()**2)
    return cost

def forward_prop(X, thetas):
    a = X
    a_arr = []

    # a = np.hstack((np.ones((len(a), 1)), a))
    # a_arr.append(a)
    # z = a.dot(thetas[0].T)
    # a = expit(z)

    a = np.hstack((np.ones((len(a), 1)), a))
    a_arr.append(a)

    z = a.dot(thetas[-1].T)
    a = np.exp(z)
    denom = np.sum(a, axis=1)
    a = a / denom.reshape(len(a), 1)

    a_arr.append(a)
    return a_arr
    
def back_prop(a_arr, y, l, thetas):
    m = float(len(y))

    dout = a_arr[-1] - y # (m, 4)
    grad_out = dout.T.dot(a_arr[-2])/m # (4, m) . (m, h+1) = (4, h+1)
    grad_out[:,1:] += l * thetas[-1][:,1:]

    # fp = np.multiply(a_arr[1][:,1:], 1 - a_arr[1][:,1:]) # (m, h)
    # dhidden = np.multiply(dout.dot(thetas[1][:,1:]), fp) # (m, 4) . (4, h) = (m, h)
    # grad_hidden = dhidden.T.dot(a_arr[0])/m # (h, m) . (m, n+1) = (h, n+1)
    # grad_hidden[:,1:] += l * thetas[0][:,1:]

    #return flat_and_concat([grad_hidden, grad_out])
    return grad_out.flatten()

def check_grad(X, y, l, sizes):
    print 'Calculating numerical gradient'
    thetas = random_thetas(sizes)
    tflat = flat_and_concat(thetas)

    def f(T, *args):
        thetas = thetas_from_flat(T, sizes)
        h = forward_prop(X, thetas)[-1]
        return cost(h, y, l, thetas)
    
    def fgrad(T, *args):
        thetas = thetas_from_flat(T, sizes)
        a_arr = forward_prop(X, thetas)	
        return back_prop(a_arr, y, l, thetas)

    return opt.check_grad(f, fgrad, tflat)

def calc_acc(X_test, y_test, thetas):
    h = forward_prop(X_test, thetas)[-1]
    pred = np.argmax(h, axis=1)
    actual = np.argmax(y_test, axis=1)
    return np.sum(pred == actual) / float(len(y_test))

def predict(X, thetas, return_h=False):
    h = forward_prop(X, thetas)[-1]
    pred = np.argmax(h, axis=1)
    if return_h:
        return h, pred
    return pred

def write_cost_and_acc():
    global Js, accs
    with open('logs/'+prog_name+'/cost_and_acc.csv', 'w') as f:
        f.write('Cost, Accuracy\n')
        for j, acc in zip(Js, accs):
            f.write('%f, %f\n' % (j, acc))

def write_last_theta():
    global last_tflat, prog_name
    np.savetxt('logs/'+prog_name+'/weights.txt', last_tflat)

def flat_and_concat(thetas):
    flat = np.array([])
    for theta in thetas:
        flat = np.append(flat, theta.flatten())
    return flat

# train a network using convergent gradient minimization
def train(X, y, l, sizes, name='test', test_set=()):
    global global_step, Js, accs, prog_name, last_tflat
    prog_name = name

    X_test, y_test = test_set
    
    thetas = random_thetas(sizes)
    tflat = flat_and_concat(thetas)
    last_tflat = tflat.copy()

    print X.shape, y.shape
    print sizes

    def f(T, *args):
        global last_tflat, Js, accs, global_step 
        global_step += 1

        last_tflat = np.copy(T)
        thetas = thetas_from_flat(T, sizes)

        a_arr = forward_prop(X, thetas)

        J = cost(a_arr[-1], y, l, thetas)
        g = back_prop(a_arr, y, l, thetas)

        if global_step % 20 == 0:
            Js.append(J)
            accs.append(calc_acc(X_test, y_test, thetas))
            print 'GS %d:' % global_step, Js[-1], accs[-1]
            write_cost_and_acc()
            write_last_theta()
            
        return J, g

    return opt.minimize(
        f, tflat, jac=True, 
        method='L-BFGS-B', 
        options={'disp': True, 'maxiter': 400}).x

