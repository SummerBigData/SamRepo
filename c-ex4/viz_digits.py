import matplotlib.pyplot as plt
import numpy as np
from random import randint
from collections import Counter
import argparse
import json

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

s = []
with open('logs/'+name+'/model.json') as f:
    s = json.loads(f.readline())['s']
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('logs/'+name+'/weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

pred = predict(X, thetas)
actual = np.argmax(y, axis=1)

rows, cols = 5, 2
_, axs = plt.subplots(rows, cols, figsize=(rows, cols))

row, col = -1, 0
for i in range(k):
    idxs = np.argwhere(actual == i).flatten()

    pred_vals = pred[idxs].flatten()
    most_comm_wrong = Counter(pred_vals.tolist()).most_common(2)[1]
    print i, most_comm_wrong

    lowest_acc = most_comm_wrong[0]
    wrongs = X[idxs][np.argwhere(pred_vals == lowest_acc).flatten()]
    x = wrongs[randint(0, len(wrongs)-1)]

    if col % cols == 0:
        row += 1
        col = 0

    dim = int(np.sqrt(len(x)))
    x = x.reshape((dim, dim))
    x *= 255
    x = 1-x

    ax = axs[row, col]
    ax.imshow(x, cmap='gray')
    ax.axis('off')

    col += 1

if args.display:
    plt.show()
