import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

import random
import sys

axs = None # global variable to keep track of the axs of the current figure

# setup the figure with <rows> rows and <cols> cols
def setup_layout(rows, cols):
	global axs
	_, axs = plt.subplots(rows, cols, figsize=(12,6))

# display just a single number
# if only one number should be plotted, don't pass shd_show or set it to True
# if idx is not passed, a random image for the given number is shown
# if shd_show=False then it is assumed that a grid of numbers is being plotted
#	and row and col should be passed to indicate the grid placement
def disp_num(n, X, shd_show=True, idx=None, row=0, col=0, title='Title'):
	if idx is None:
		first_idx = 500 * n
		idx = random.randint(first_idx, first_idx+499)
	x = X[idx].reshape((20,20)).T * 255	

	if shd_show:
		plt.imshow(x, cmap='gray')
		plt.show()
	else:
		global axs
		ax = axs[row, col]
		ax.imshow(x, cmap='gray')	
		ax.set_title(title, fontsize=10)
		ax.axis('off')

# display all the numbers in a grid
def disp_nums(X):
	num = 0
	setup_layout(2, 5) # create the layout with 2 rows and 5 columns
	for i in range(2):
		for j in range(5):
			disp_num(num, X, shd_show=False, row=i, col=j, title='Example %d' % (num))
			num += 1
	show()

# show the final plot
def show():
	# tight_layout needed for plots with multiple subplots
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Please pass a number of the image to display'
		exit()
	num_to_disp = int(sys.argv[1])

	mat = loadmat('data/data.mat')
	X = mat['X']
	
	# if the number to display is less than 0, display all the numbers in a grid
	# probably preferable to pass a command line flag but I didn't feel like it
	if num_to_disp < 0:
		disp_nums(X)
	else:
		disp_num(num_to_disp, X)
