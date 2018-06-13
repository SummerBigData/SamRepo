from scipy.io import loadmat
import numpy as np

from nn_funcs import *

mat = loadmat('data/weights.mat')
t1 = mat['Theta1']
t2 = mat['Theta2']

mat = loadmat('data/data.mat')
X = mat['X']
y = mat['y']

h = feed_forward(X, [t1, t2])
acc = calc_acc(h, y)
print 'Neural network has %f percent accuracy.' % (acc) 
