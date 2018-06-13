import cv2
import numpy as np
import json
import argparse
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.interpolation import shift
import atexit
import uuid
import os

from nn_util import *

parser = argparse.ArgumentParser(description='Neural Network backpropagation')
parser.add_argument('--all_data', dest='use_all_data', action='store_true', help='Whether or not to use to full dataset')
parser.add_argument('--shd_split', dest='split', action='store_true', help='Whether or not to split the dataset into training and testing sets')
parser.add_argument('--name', dest='name', type=str, default='nn', help='help ya dam self')
parser.add_argument('--shd_disp', dest='display', action='store_true', help='Whether or not to display the confidence histograms for every digit')
args = parser.parse_args()
name = args.name

if args.split:
	_, _, X, y = load_data(all_data=args.use_all_data, train=False, split=True)
else:
	X, y = load_data(all_data=args.use_all_data, train=False)
m, n = X.shape[0], X.shape[1]
k = 10
actual = np.argmax(y, axis=1)

s = []
with open('logs/'+name+'/model.json') as f:
	s = json.loads(f.readline())['s']
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('logs/'+name+'/weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

drawing = False
prev = (None, None)
curr_pred = None

num_corr = 0
num_total = 0

def save_imgs():
    global xs, ys
    if xs is None:
        return
    uid = str(uuid.uuid4())
    os.mkdir('drawn_images/'+uid)
    np.savetxt('drawn_images/'+uid+'/X.txt', xs)
    np.savetxt('drawn_images/'+uid+'/y.txt', ys)

atexit.register(save_imgs)

def print_img(x):
    for i, row in enumerate(x):
        if np.sum(row) == 255 * 28:
            continue
        for j, datum in enumerate(row):
            #if datum == 0:
                #x[i,j] += max(0, np.random.random_sample() - 0.8) * 255
            #print x[i,j],
            if datum == 255:
                print ' ',
                continue
            print datum,
        print ''
    print '\n///\n'

xs = None
ys = None

def classify(n):
    if drawing:
        return

    x = img[:,:,0]
    # vmean = np.mean(1.0 - x/255.0, axis=0)
    # hmean = np.mean(1.0 - x/255.0, axis=1)
    # idxs = np.array(list(range(280)))
    # vidx = int(np.dot(vmean, idxs) / np.sum(vmean))
    # hidx = int(np.dot(hmean, idxs) / np.sum(hmean))

    # dx = 139.5-hidx
    # dy = 139.5-vidx

    # M = np.float32([[1, 0, dx], [0, 1, dy]])
    # x = cv2.warpAffine(x, M, x.shape, borderValue=255.0)

    x = cv2.resize(x, (28, 28), cv2.INTER_NEAREST)
    x = gaussian_filter(x, sigma=0.5, truncate=0.75)
    for i in range(x.shape[0]):
        for j, d in enumerate(x[i]):
            if d == 0:
                x[i,j] += int(np.random.random_sample() * 25)
    #print_img(x)
    
    x = 1.0 - x.flatten()/255.0

    global xs, ys
    if xs is None:
        xs = np.array([x])
        ys = np.array([n], dtype=float)
    else:
        xs = np.append(xs, np.array([x]))
        ys = np.append(ys, np.array([n], dtype=float))

    #print_img(x.reshape((28, 28)))
    cv2.imshow('result', x.reshape((28, 28)))

    X_test = np.array(x).reshape((1, x.shape[0]))
    h, prediction = predict(X_test, thetas, return_h=True)
    prediction = prediction[0]

    idxs = np.argwhere(actual == prediction).flatten()
    test_img = X[idxs][np.random.randint(0, len(idxs))].reshape((28, 28))
    #print_img(((1.0 - test_img)*255.0).astype(np.uint8))
    cv2.imshow('example', test_img)

    print "Prediction: ", prediction
    print "Confidence: ", h[0, prediction]
    if prediction != n:
        print "Correct Confidence: ", h[0, n]

    global num_total, num_corr
    num_total += 1
    if n == prediction:
        num_corr += 1
    print '%f percent accuracy' % (num_corr/float(num_total))
    print '\n***\n'

    curr_pred = prediction
    

def draw(event, x, y, flags, param):
    global drawing, prev

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev = (x, y)

    elif drawing and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, prev, (x, y), (0, 0, 0), 20)
        prev = (x, y)

    elif drawing and event == cv2.EVENT_LBUTTONUP:
        if (x, y) == prev:
            cv2.circle(img, prev, 5, (0, 0, 0), -1)

        drawing = False
        prev = (None, None)
    

def initialize_image():
    img = np.ones((280, 280, 1), np.uint8)
    return img * 255

img = initialize_image()

cv2.namedWindow('canvas')
cv2.setMouseCallback('canvas', draw)

while(1):
    cv2.imshow('canvas', img)
    k = cv2.waitKey(20)
    if k == 27:
        break
    elif k >= 48 and k < 58: # 0-9
        classify(k - 48)
    elif k == 99: # c
        img = initialize_image()
        prev = (None, None)
        drawing = False
        training = False
        cv2.imshow('canvas', img)

cv2.destroyAllWindows()
