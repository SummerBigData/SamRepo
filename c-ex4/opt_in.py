import numpy as np
import random
import json
import argparse

import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

from nn_util import *

parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--name', dest='name', type=str, default='nn', help='help ya dam self')
args = parser.parse_args()
name = args.name

k = 10
m = 10000
X_data = np.random.uniform(size=(m, 784))
y_data = np.random.random_integers(0, k-1, size=m)
y_data = onehot(y_data)
print X_data.shape, y_data.shape

m, n = X_data.shape[0], X_data.shape[1]

s = []
with open('logs/'+name+'/model.json') as f:
	s = json.loads(f.readline())['s']
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('logs/'+name+'/weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

X = tf.placeholder('float64', [None, 784])
y = tf.placeholder('float64', [None, 10])

weights = {
    'W1': tf.Variable(thetas[0][:,1:]),
    'b1': tf.Variable(thetas[0][:,1]),
    'W2': tf.Variable(thetas[1][:,1:]),
    'b2': tf.Variable(thetas[1][:,1]),
    'W3': tf.Variable(thetas[2][:,1:]),
    'b3': tf.Variable(thetas[2][:,1])
}

h1 = tf.nn.sigmoid(tf.matmul(X,  tf.transpose(weights['W1'])) + weights['b1'])
h2 = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(weights['W2'])) + weights['b2'])
logits = tf.matmul(h2, tf.transpose(weights['W3']) + weights['b3'])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
grads = tf.gradients(cost, X)[0]

def loss(x, *args):
    sess, y_batch = args
    retval = sess.run([h1, h2, logits, cost], feed_dict={
        X: x.reshape(-1, 784),
        y: y_batch
    })[-1]
    return retval

def grad(x, *args):
    sess, y_batch = args
    retval = sess.run([h1, h2, logits, cost, grads], feed_dict={
        X: x.reshape(-1, 784),
        y: y_batch
    })[-1].flatten()
    return retval

print y_data.shape
init = tf.global_variables_initializer()
print y_data.shape
b_size = 100

actual = np.argmax(y_data, axis=1)
print actual[:10]
print np.argwhere(actual == 0).flatten()[:10]

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(20):
        num_batches = m//b_size

        #batches = np.hstack((X_data, actual.reshape(-1, 1)))
        #np.random.shuffle(batches)

        #X_batches = batches[:,:-1]
        #y_batches = batches[:,-1]
        #y_batches = onehot(y_batches)

        for i in range(num_batches):
            print 'Epoch %d: Batch %d/%d' % (epoch, i, num_batches)
            X_batch, y_batch = X_data[i*b_size:(i+1)*b_size], y_data[i*b_size:(i+1)*b_size]
            X_batch = fmin_l_bfgs_b(loss, X_batch.flatten(), fprime=grad, args=(sess, y_batch), maxiter=200)[0]
            X_data[i*b_size:(i+1)*b_size] = X_batch.reshape(-1, 784)

    print 'Saving inputs'
    np.savetxt('X_learned2.txt', X_data)

import matplotlib.pyplot as plt
from scipy.special import expit

for i in range(k):
    print i
    idxs = np.argwhere(actual == i).flatten()
    idx = np.random.choice(idxs)

    a = np.hstack((np.ones((1, 1)), X_data[idx].reshape(1, 784)))
    h1 = expit(a.dot(thetas[0].T))
    h1 = np.hstack((np.ones((1, 1)), h1))
    h2 = expit(h1.dot(thetas[1].T))
    h2 = np.hstack((np.ones((1, 1)), h2))
    out = expit(h2.dot(thetas[2].T))
    print out[0]
    print out[0,i]

    plt.imshow(X_data[idx].reshape(28, 28), cmap='gray')
    plt.show()
