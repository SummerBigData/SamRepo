import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift

def crop(X):
	return X[:,4:-4]

def random_rotation(img, angle_range=(-35, 35)):
    dim = int(np.sqrt(len(img)))
    img = img.reshape((dim, dim))
    angle = np.random.randint(*angle_range)
    img = rotate(img, angle, order=0)
    return imresize(img, (dim, dim)).flatten()/255.0

def random_scale(img):
    dim = int(np.sqrt(len(img)))
    img = img.reshape((dim, dim))

    scale = max(np.random.random_sample() + 0.2, 0.8)
    new_dim = int(dim*scale)
    while new_dim == dim:
        scale = max(np.random.random_sample() + 0.2, 0.8)
        new_dim = int(dim*scale)

    img = imresize(img, (new_dim, new_dim))
	
    diff = abs(new_dim - dim)//2
    if not diff % 2 == 0 or abs(new_dim - dim) == 1:
        diff += 1

    if scale > 1.0:
    	img = img[diff:-diff, diff:-diff]
    else:
        img = np.pad(img, pad_width=diff, mode='constant', constant_values=0.0)
    
    return imresize(img, (dim, dim)).flatten()/255.0

def random_translation(img):
    dim = int(np.sqrt(len(img)))
    img = img.reshape((dim, dim))
    xShift, yShift = (np.random.random_sample(2) * 2.0 - 1.0) * (np.random.random_sample(1) * 5.0)
    return shift(img, (xShift, yShift), order=0).flatten()

def random_stretch(img, axis=0):
    dim = int(np.sqrt(len(img)))
    img = img.reshape((dim, dim))

    scale = np.random.random_sample()/3.0 + 0.33
    new_dim = int(dim*scale)
    while new_dim == dim:
        scale = np.random.random_sample()/3.0 + 0.33
        new_dim = int(dim*scale)

    h, w = (new_dim, new_dim)
    if axis == 0:
        h = dim
    else:
        w = dim
    print h, w
    img = imresize(img, (h, w))

    diff = abs(new_dim - dim)//2
    if not diff % 2 == 0 or abs(new_dim - dim) == 1:
        diff += 1

    img = np.pad(img, pad_width=diff, mode='constant', constant_values=0)
    return imresize(img, (dim, dim)).flatten()/255.0

def augment(X, y, ds_dig, aug_mode):
    from nn_util import shuffle

    actual = np.argmax(y, axis=1)
    dig_idxs = np.argwhere(actual == ds_dig).flatten()
    idx_to_del = np.random.choice(dig_idxs, 9*len(dig_idxs)//10, replace=False)
    X = np.delete(X, idx_to_del, axis=0)
    y = np.delete(y, idx_to_del, axis=0)
    actual = np.argmax(y, axis=1)
    
    xs = X[np.argwhere(actual == ds_dig).flatten()].copy()
    xs = np.tile(xs, (10, 1))
    ys = np.array([float(ds_dig)] * len(xs))
    
    def aug(x):
        if aug_mode == 1:
            return x
        elif aug_mode == 2:
            return random_rotation(x)
        elif aug_mode == 3:
            return random_scale(x)
        elif aug_mode == 4:
            return random_translation(x)
        elif aug_mode == 5:
            return random_rotation(random_scale(x))
        elif aug_mode == 6:
            return random_translation(random_rotation(x))
        elif aug_mode == 7:
            return random_translation(random_scale(x))
        else:
            return random_translation(random_rotation(random_scale(x)))

    xs = np.array(list(map(aug, xs)))
    xs, ys = shuffle(xs, ys)
    tmp_ys = np.zeros((len(ys), 10))
    tmp_ys[:,int(ys[0])] = 1.0
    ys = tmp_ys

    X = np.append(X, xs, axis=0)
    y = np.append(y, ys, axis=0)
    return X, y

if __name__ == '__main__':
    from display import *
    from nn_util import load_data

    X, y = load_data(all_data=True)
    x = X[np.random.randint(0, len(X))]
    display_img(x)

    for _ in range(5):
        #display_img(random_rotation(x))
        #display_img(random_scale(x))
        #display_img(random_rotation(random_scale(x)))
        #display_img(random_stretch(x))
        #display_img(random_stretch(x, axis=1))
        #display_img(random_translation(x))
        display_img(random_translation(random_rotation(random_scale(x))))