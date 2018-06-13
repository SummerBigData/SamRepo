import cv2
import numpy as np
from numpy.polynomial import polynomial as P
from math import pi, atan
import json
from scipy.misc import imresize
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.interpolation import shift
import uuid
import os

from nn_util import *

drawing = False
prev = (None, None)
curr_pred = None

# X, y = load_data(all_data=True, train=False)
# m, n = X.shape[0], X.shape[1]
# k = 10
# actual = np.argmax(y, axis=1)

# sizes = [(50, 785), (25, 51), (10, 26)]
# theta_flat = np.genfromtxt('logs/h50_L4_l3_s60000/weights.txt')
# thetas = thetas_from_flat(theta_flat, sizes)

from sae_util import weights_from_flat
tflat = np.genfromtxt('../dl-stl/logs/s10k/weights_sae.txt')
sae_sizes = [(196, 784), (196, 1)]
Ws, bs = weights_from_flat(tflat, sae_sizes)

sizes = [(25, 197), (10, 26)]
theta_flat = np.genfromtxt('../dl-stl/logs/s10k/weights_nn.txt')
thetas = thetas_from_flat(theta_flat, sizes)

def predict_stl(X, thetas, Ws, bs, return_h=False):
    import sae_util
    X = sae_util.forward_prop(X, Ws, bs)[0][1]
    h = forward_prop(X, thetas)[-1]
    pred = np.argmax(h, axis=1)
    if return_h:
        return (h, pred)
    return pred

def classify(n, x):
    if drawing:
        return

    x = cv2.resize(x, (28, 28), cv2.INTER_NEAREST)
    x = gaussian_filter(x, sigma=0.5, truncate=0.75)
    for i in range(x.shape[0]):
        for j, d in enumerate(x[i]):
            if d == 0:
                x[i,j] += int(np.random.random_sample() * 25)
    
    x = 1.0 - x.flatten()/255.0
    #cv2.imshow('result %d' % n, x.reshape((28, 28)))

    X_test = np.array(x).reshape((1, x.shape[0]))
    #h, prediction = predict(X_test, thetas, return_h=True)
    h, prediction = predict_stl(X_test, thetas, Ws, bs, return_h=True)
    prediction = prediction[0]

    # idxs = np.argwhere(actual == prediction).flatten()
    # test_img = X[idxs][np.random.randint(0, len(idxs))].reshape((28, 28))
    #cv2.imshow('example', test_img)

    # print "Prediction: ", prediction
    # print "Confidence: ", h[0, prediction]
    # if prediction != n:
    #     print "Correct Confidence: ", h[0, n]
    return prediction
    

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
    img = np.ones((280, 280*3 + 10, 1), np.uint8)
    return img * 255

img = initialize_image()

cv2.namedWindow('canvas')
cv2.setMouseCallback('canvas', draw)

def align(x):
    s = x.shape[0]

    index = np.zeros(x.shape[1])
    for i in range(s):
        index[i] = i + 1

    mat = x
    hcenter = np.zeros(x.shape[1])
    
    for j in range(s):
        
        if sum(mat[j]) == 0:
            hcenter[j] = -1
        
        else:
            hcenter[j] = sum(mat[j]*index)/ (sum(mat[j])+0.0)
    
    weights = np.zeros(x.shape[1])
    for j in range(s):
        if hcenter[j] < 0:
            weights[j] = 0
        else:
            weights[j] = 1

    c = P.polyfit(index,hcenter,1,full=False, w=weights)
    return rotate(mat, -(-1*atan(c[1])*180.0/pi), reshape=False, cval=255.0)

def split():
    global img
    x = img[:,:,0]

    boundaries = []
    curr_bound = None

    i = 0
    while i < x.shape[1]:
        if np.sum(x[:,i:i+5]) == 255*x.shape[0]*5:
            x[:,i:i+5] = 0
        i += 5

    i = 0
    while i < x.shape[1]:
        if curr_bound is None:
            while i < x.shape[1] and np.sum(x[:,i]) == 0:
                i += 1
        else:
            while i < x.shape[1] and np.sum(x[:,i]) > 0:
                i += 1

        if curr_bound is None:
            curr_bound = i
        else:
            boundaries.append((curr_bound, i))
            curr_bound = None

        i += 1

    for bound in boundaries:
        j = 0
        while j < x.shape[0]:
            if np.sum(x[j:j+5,bound[0]:bound[1]]) == 255*(bound[1]-bound[0])*5:
                x[j:j+5,bound[0]:bound[1]] = 0
            j += 5

    images = []
    for i, boundary in enumerate(boundaries):
        sub_img = x[:,boundary[0]:boundary[1]]
        y_mask = np.where(sub_img > 0)
        y_min, y_max = np.min(y_mask[0]), np.max(y_mask[0])
        sub_img = sub_img[y_min:y_max]

        a = np.where(sub_img < 255)
        minX, maxX = np.min(a[1]), np.max(a[1])
        minY, maxY = np.min(a[0]), np.max(a[0])

        if minY > sub_img.shape[0] - maxY:
            y_pad = (0, minY - (sub_img.shape[0] - maxY))
        else:
            y_pad = ((sub_img.shape[0] - maxY) - minY, 0)

        if minX > sub_img.shape[1] - maxX:
            x_pad = (0, minX - (sub_img.shape[1] - maxX))
        else:
            x_pad = ((sub_img.shape[1] - maxX) - minX, 0)

        pad_width = [y_pad, x_pad]
        sub_img = np.pad(sub_img, pad_width, 'constant', constant_values=255.0)

        diff = sub_img.shape[0] - sub_img.shape[1]
        if diff < 0:
            sub_img = np.pad(sub_img, [(-diff//2, -diff//2), (0,0)], 'constant', constant_values=255.0)
        elif diff > 0:
            sub_img = np.pad(sub_img, [(0,0), (diff//2, diff//2)], 'constant', constant_values=255.0)

        final_pad = 22
        sub_img = np.pad(sub_img, [(final_pad, final_pad), (final_pad, final_pad)], 'constant', constant_values=255.0)
        images.append(sub_img)

    final_num = 0
    for i, num in enumerate(images):
        #n = int(input())
        sub_num = classify(0, num)
        final_num *= 10
        final_num += sub_num
    print final_num
    

while(1):
    cv2.imshow('canvas', img)
    k = cv2.waitKey(20)
    if k == 27:
        break
    elif k >= 48 and k < 58: # 0-9
        split()
    elif k == 99: # c
        img = initialize_image()
        prev = (None, None)
        drawing = False
        training = False
        cv2.imshow('canvas', img)

cv2.destroyAllWindows()
