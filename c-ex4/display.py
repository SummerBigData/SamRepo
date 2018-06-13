import matplotlib.pyplot as plt
import numpy as np
from random import randint

def display_tests(X, y):
    for i in range(2):
        idx = randint(0, len(X))
        print y[idx]
        display_img(1 - X[idx])

def display_img(img):
    dim = int(np.sqrt(len(img)))
    img *= 255
    img = 1-img
    plt.imshow(img.reshape((dim, dim)), cmap='gray')
    plt.show()
