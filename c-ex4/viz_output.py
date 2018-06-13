import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from nn_util import *

n = 400
k = 10
s = [n, 25, k]
L = len(s)

multiplier = 10
X = np.random.random((k*multiplier, n))
#y = np.diag(np.ones(k))
y = np.array([])
for i in range(k):
	y = np.append(y, np.array([i] * multiplier))

onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y.reshape((len(y), 1))) # onehot encode y

sizes = [(s[i+1], s[i]+1) for i in range(L-1)]
theta_flat = np.genfromtxt('weights.txt')
thetas = thetas_from_flat(theta_flat, sizes)

print check_grad_train_X(X, y, k, n, k, L, sizes, thetas)

"""
X = train_train_X(X, y, k, n, k, L, sizes, thetas).reshape((k * multiplier, n))
new_X = np.zeros((k, n))
for i in range(k):
	new_X[i] = X[i*multiplier]
X = new_X

x = X[0].reshape((1, n))
print forward_prop(x, thetas)[-1]

x = X[1].reshape((1, n))
print forward_prop(x, thetas)[-1]

rows, cols = 5, 2
_, axs = plt.subplots(rows, cols, figsize=(rows, cols))

row, col = -1, 0
for x in X:
	if col % cols == 0:
		row += 1
		col = 0

	x = x.reshape((20, 20))
	ax = axs[row, col]
	ax.imshow(x, cmap='gray')
	ax.axis('off')

	col += 1

plt.show()
"""
