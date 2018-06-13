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
X_data = np.random.randn(10000, 784)
y_data = np.random.random_integers(0, k-1, size=10000)
y_data = onehot(y_data)

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
    retval = sess.run(cost, feed_dict={
        X: x.reshape(-1, 784),
        y: y_batch
    })
    return retval

def grad(x, *args):
    sess, y_batch = args
    retval = sess.run(grads, feed_dict={
        X: x.reshape(-1, 784),
        y: y_batch
    }).flatten()
    return retval

init = tf.global_variables_initializer()
b_size = 250

actual = np.argmax(y_data, axis=1)
print actual[:10]

print np.argwhere(actual == 0)[:10]

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1):
        num_batches = m//b_size
        for i in range(num_batches):
            print 'Epoch %d: Batch %d/%d' % (epoch, i, num_batches)

            X_batch, y_batch = X_data[i*b_size:(i+1)*b_size], y_data[i*b_size:(i+1)*b_size]
            X_batch = fmin_l_bfgs_b(loss, X_batch.flatten(), fprime=grad, args=(sess, y_batch), maxiter=200)[0]
            X_data[i*b_size:(i+1)*b_size] = X_batch.reshape(-1, 784)

    np.savetxt('X_learned.txt', X_data)

import matplotlib.pyplot as plt
for i in range(k):
    idxs = np.argwhere(actual == i).flatten()
    idx = np.random.choice(idxs)
    plt.imshow(X_data[idx].reshape(28, 28), cmap='gray')
    plt.show()