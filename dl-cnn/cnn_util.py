from scipy.io import loadmat
from scipy.special import expit
from scipy.signal import convolve2d as conv2d
import numpy as np

import os
from nn_util import onehot

def load_data(train=True, conv=False, path=None, shuffle=True):
    if train:
        filename = 'train.mat'
        xkey, ykey = 'trainImages', 'trainLabels'
    else:
        filename = 'test.mat'
        xkey, ykey = 'testImages', 'testLabels'

    mat = loadmat('data/'+filename)
    X, y = mat[xkey], mat[ykey]

    if not conv:
        # the dataset is loaded for some reason such that the set index is last
        # transpose the dataset so this is not so
        X = np.transpose(X, (3, 0, 1, 2))
    else:
        filename = path + '/X_conv_train.txt'
        if not train:
            filename = path + '/X_conv_test.txt'
        X = np.genfromtxt(filename)

    y = onehot(y)

    if shuffle:
        idxs = np.array(list(range(len(X))))
        idxs = np.random.choice(idxs, size=len(idxs), replace=False)
        return X[idxs], y[idxs]
    else:
        return X, y

def zca(X):
    # mean normalize the dataset
    mu = np.mean(X, axis=0).reshape(1, X.shape[1])
    X -= np.tile(mu, (X.shape[0], 1))

    # calculate the covariance matrix
    Sigma = np.dot(X.T, X)/X.shape[0]

    # calculate the eigenvalues and eigenvectors of the cov. matrix
    U, S, _ = np.linalg.svd(Sigma)
    S = S.reshape(len(S), 1)

    # compute the ZCA whitening and multiply the dataset by it
    epsilon = 0.1
    l_mat = np.diag(1.0/np.sqrt(S.flatten() + epsilon))
    zca_mat = U.dot(l_mat).dot(U.T)
    return X.dot(zca_mat), zca_mat


from time import time
import multiprocessing
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def convolve_and_pool(X, W, b, Z, mu, shd_pool=True, num_proc=4):
    # First, divide the dataset into 20 batches
    #
    # Then on each group of 4 batches, create four separate processes
    # to convolute and pool each batch in parallel. This greatly improves the function speed
    #
    # By pooling each batch directly after convolution, we also greatly
    # reduce the amount of memory needed for convolution.
    # If we were to convolve on the entire dataset, we would easily run into a MemoryError

    batches = num_proc * 5
    assert len(X) % batches == 0, 'm must be divisible by %d' % batches

    print 'Convolving and pooling'
    start = time()

    batch_size = len(X)//batches
    out_dim = (X.shape[2]-7)//19
    print 'BATCH SIZE:', batch_size

    proc_func = conv_pool_unpack
    if not shd_pool:
        proc_func = conv_unpack
        out_dim = X.shape[2]-7

    res = np.zeros((X.shape[0], W.shape[0], out_dim, out_dim)) 
    print res.shape
    num_batch = num_proc

    for i in range(0, batches, num_batch):
        print 'Batch Group: %d/%d' % (i//num_batch+1, num_batch+1)

        with poolcontext(processes=num_batch) as p:
            batches = [X[k*batch_size:(k+1)*batch_size] for k in range(i, i+num_batch)]
            batches = [(batch, W, b, Z, mu) for batch in batches]
            res_batches = p.map(proc_func, batches)

        pids = [rb[0] for rb in res_batches]
        for j in range(len(pids)-1):
            if pids[j] > pids[j+1]:
                print 'ERROR: pids out of order'
                print pids
                res_batches = sorted(res_batches, key=lambda tup: tup[0])

        res_batches = [rb[1] for rb in res_batches]
    
        s = i*batch_size
        for k in range(len(res_batches)):
            res[s + k*batch_size: s + (k+1)*batch_size] = res_batches[k]

    print 'Convolve and pool completed in %d seconds\n' % int(time() - start)
    return res

def conv_unpack(args):
    return convolve(*args)

def conv_pool_unpack(args):
    pid, X = convolve(*args)
    return (pid, pool(X))

def convolve(X, W, b, Z, mu):
    # W is 400 by 192 (8x8x3)
    # this means that we have 400 features to learn from the 64x64x3 images
    # 
    # Convolving one channel of the 64x64 leads to a 57x57 pixel image
    # therefore after convolution, X.shape = (m 400 57 57)
    #
    # After convolution, flatten res so that res.shape = (m 400*57*57)
    # and tile b (which has shape (400 1)) so that it can be added elementwise to res

    # Since the training for W and b was done on ZCA-whitened images, we must instead
    # of using W.xx + b in convolution, we must use W.(Z.(x-mu)) + b = W.Z.x + b - W.Z.mu
    #
    # Z.shape = (192 192) so W.dot(Z).shape = (400 192) = W.shape
    # mu.shape = (192 1) so W.Z.mu.shape = (400 1) = b.shape
    #print 'Convolving %d images' % X.shape[0]
    start = time()

    WT = W.dot(Z)
    bm = b - WT.dot(mu)

    dim = int(np.sqrt(W.shape[1]/3.0)) # 8
    out_size = X.shape[1] - dim + 1 # 57
    num_chan = X.shape[3] # 3
    num_img = X.shape[0] # m - 8 for testing, 2000 for training
    res = np.zeros((num_img, W.shape[0], out_size, out_size)) # mx400x57x57

    for k, x in enumerate(X):
        for j, w in enumerate(WT):
            conv_img = np.zeros((out_size, out_size), dtype=np.float64)
            for i in range(num_chan):
                xchan, wchan = x[:,:,i], w[i*dim**2 : (i+1)*dim**2].reshape(dim, dim)
                wchan = np.flipud(np.fliplr(wchan))
                conv_img += conv2d(xchan, wchan, mode='valid')
            conv_img += bm[j]
            res[k,j,:,:] = expit(conv_img)

    #print 'Convolution finished in %d seconds' % int(time() - start)
    return (os.getpid(), res)

def test_convolution(X, X_conv, W, b, Z, mu):
    # Test the convolve function by picking 1000 random patches from the input data,
    # preprocessing it using Z and mu, and feeding it through the SAE (using W and b)
    #
    # If the result is close to the convolved patch, we're good

    patch_dim = int(np.sqrt(W.shape[1]/3.0)) # 8
    conv_dim = X.shape[1] - patch_dim + 1 # 57

    for i in range(100):
        feat_no = np.random.randint(0, W.shape[0])
        img_no = np.random.randint(0, X.shape[0])
        img_row = np.random.randint(0, conv_dim)
        img_col = np.random.randint(0, conv_dim)

        patch_x = (img_col, img_col+patch_dim)
        patch_y = (img_row, img_row+patch_dim)

        # Obtain a 8x8x3 patch and flatten it to length 192
        patch = X[img_no, patch_y[0]:patch_y[1], patch_x[0]:patch_x[1]]
        patch = np.concatenate((patch[:,:,0].flatten(), patch[:,:,1].flatten(), patch[:,:,2].flatten())).reshape(-1, 1)
        #patch = patch.reshape(-1, 1)

        # Preprocess the patch
        patch -= mu
        patch = Z.dot(patch) 

        # Feed the patch through the autoencoder weights
        # now sae_patch.shape = (400 192) . (192 1) = (400 1)
        sae_feat = expit(W.dot(patch) + b)

        # Compare it to the convolved patch
        conv_feat = X_conv[img_no,:,img_row,img_col]
        #print conv_feat.reshape(20, 20)

        err = abs(sae_feat[feat_no, 0] - conv_feat[feat_no])
        """
        import matplotlib.pyplot as plt
        img = np.zeros((20, 42))
        img[:,:20] = sae_feat.reshape(20, 20)
        img[:,22:] = conv_feat.reshape(20, 20)
        plt.imshow(img, cmap='gray')
        plt.show()
        """
        if i == 5:
            exit()
        print err

def pool(X, wsize=19):
    # Perform mean pooling on the convolved features with a default window size of 19.
    #
    # Since each convolved feature has shape 57x57 and the window size is 19,
    # the resulting feature shape after pooling will be 3x3 (57/19 = 3)
    #
    # Assume that the window_size passed is a divisor of the convolved feature length

    num_slides = X.shape[2] / wsize
    res = np.zeros((X.shape[0], X.shape[1], num_slides, num_slides))
    for i in range(num_slides):
        for j in range(num_slides):
            window = X[:, :, i*wsize:(i+1)*wsize, j*wsize:(j+1)*wsize]
            res[:,:,i,j] = np.mean(window, axis=(2,3))
    return res
