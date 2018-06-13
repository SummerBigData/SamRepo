import numpy as np
import argparse
import json
import atexit
import os

from cnn_util import *
from softmax_util import check_grad, train

# add hella arguments
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--check_grad', dest='check_grad', action='store_true', help='Whether or not to numerical check the gradient from backpropagation (very expensive)')
parser.add_argument('--check_conv', dest='check_conv', action='store_true', help='Whether or not to check if convolution was correctly implemented')
parser.add_argument('--check_W', dest='checkw', action='store_true', help='Whether or not to visualize the saved weights from the linear SAE')
parser.add_argument('--name', dest='name', help='The name of the program (for logs and plots)', type=str, default='test')
parser.add_argument('-m', dest='m', help='How many to sample', type=int, default=2000)
args = parser.parse_args()

if not os.path.isdir('logs/'+args.name):
	os.mkdir('logs/'+args.name)


train_conv_exists = os.path.isfile('X_conv_train.txt')
test_conv_exists = os.path.isfile('X_conv_test.txt')


if not train_conv_exists:
    X, y = load_data()
    img_len = X.shape[1] * X.shape[2] * X.shape[3]
    zcaw = np.genfromtxt('zca_sae.txt')
    mu = np.genfromtxt('mu_sae.txt')
    mu = mu.reshape(len(mu), 1)

    X, y = X[:args.m], y[:args.m]
    m, n = X.shape[0], img_len
    h = n
    with open('model_lsae.json') as f:
        h = json.loads(f.readline())['s'][1]

    k = 4 # number of classes

    tflat = np.genfromtxt('weights_lsae.txt')
    wlen = X.shape[1] * X.shape[3]
    W = tflat[:wlen*h].reshape(h, wlen)
    b = tflat[wlen*h:wlen*h+h].reshape(h, 1)

    if args.checkw:
        from disp import display_encoding
        display_encoding(W.dot(zcaw))
        exit()

    shd_pool = not args.check_conv
    X_conv = convolve_and_pool(X, W, b, zcaw, mu, shd_pool=shd_pool)
    #X_conv = convolve(X, W, b, zcaw, mu)[1]
    #print np.linalg.norm(X_conv - X_conv2)/np.linalg.norm(X_conv + X_conv2)
    #print np.sum(X_conv == X_conv2)
    #exit()

    if args.check_conv:
        test_convolution(X, X_conv, W, b, zcaw, mu)
        exit()

    #X_conv = pool(X_conv)
    X = X_conv.reshape(m, -1)
else:
    X, y = load_data(conv=True, path='.')    

k = 4
#h = 250
#sizes = [(h, X.shape[1]+1), (h//10, h+1), (k, h//10+1)]
#sizes = [(h, X.shape[1]+1), (k, h+1)]
sizes = [(k, X.shape[1]+1)]

if args.check_grad:
    m = min(len(X), 100)
    X = X[:m]
    y = y[:m]

    h = 5
    sizes = [(h, X.shape[1]+1), (k, h+1)]
    #sizes = [(k, X.shape[1]+1)]

    err = check_grad(X, y, 1e-4, sizes)
    print err
    exit()


if not test_conv_exists:
    X_test, y_test = load_data(train=False)
    #X_test, y_test = X_test[:test_num], y_test[:test_num]
    X_test = convolve_and_pool(X_test, W, b, zcaw, mu).reshape(len(X_test), -1)
    #X_test = pool(convolve(X_test, W, b, zcaw, mu)[1]).reshape(test_num, -1)
else:
    X_test, y_test = load_data(train=False, conv=True, path='.')

with open('logs/'+args.name+'/model.json', 'w') as f:
    f.write(json.dumps({"sizes": sizes}))

theta_flat = train(X, y, 1e-4, sizes, name=args.name, test_set=(X_test, y_test))
np.savetxt('logs/'+args.name+'/weights.txt', theta_flat)
