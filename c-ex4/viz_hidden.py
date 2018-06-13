import matplotlib.pyplot as plt
import numpy as np

from nn_util import *

n = 400
k = 10
s = [n, 25, k]
L = len(s)

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)
theta_h = thetas[0][:,1:]

def inv_sigmoid(a):
	return np.log(a/(1-a))

small_number = 1e-8
active_val = 1 - small_number

X = np.zeros((25, n))
for i in range(s[1]):
	a = np.array(s[1] * [small_number]).reshape((1, s[1]))
	a[0,i] = active_val
	z = inv_sigmoid(a)
	x = z.dot(np.linalg.pinv(theta_h).T)
	X[i] = x

rows, cols = 5, 5
_, axs = plt.subplots(rows, cols, figsize=(rows, cols))

row, col = -1, 0
for x in X:
	if col % cols == 0:
		row += 1
		col = 0

	x = x.reshape((20, 20)).T
	ax = axs[row, col]
	ax.imshow(x, cmap='gray')
	ax.axis('off')

	col += 1

plt.show()
