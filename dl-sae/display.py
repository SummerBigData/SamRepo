import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def display_sample(X, samp_size=100):
    dim = int(np.sqrt(X.shape[1]))

    idxs = list(range(len(X)))
    idxs = np.random.choice(idxs, samp_size, replace=False)
    samp = X[idxs]

    grid_dim = int(np.sqrt(samp_size))
    padding = 2
    w, h = grid_dim*dim + padding*(grid_dim+1), grid_dim*dim + padding*(grid_dim+1)
    row, col = -1, -1
    grid = np.zeros((h, w))
    #grid -= 1

    for s in samp:
        col += 1
        if col % grid_dim == 0:
            col = 0
            row += 1
    
        x_left = dim*col + (col+1)*padding
        x_right = dim*(col+1) + (col+1)*padding
        y_top = dim*row + (row+1)*padding
        y_bottom = dim*(row+1) + (row+1)*padding
        
        grid[y_top:y_bottom, x_left:x_right] = s.reshape(dim, dim)

    #plt.imshow(grid, cmap='gray', vmin=-1.0, vmax=1.0)
    plt.imshow(grid, cmap='gray', interpolation='none')
    plt.show()

def display_encoding(W, samp_size=25):
    dim = int(np.sqrt(W.shape[1]))

    X = np.zeros(W.shape)
    for i in range(W.shape[0]):
        X[i] = W[i,:] / np.sqrt(np.sum(W[i,:]**2))
    
    # sigma = np.std(X, axis=1).reshape(X.shape[0], 1)
    # sigma = np.tile(sigma, (1, X.shape[1]))
    # mu = np.mean(X, axis=1).reshape(X.shape[0], 1)
    # mu = np.tile(mu, (1, X.shape[1]))
    # X = (X-mu)/sigma

    X -= np.mean(X)
    sigma = 3*np.std(X) # truncate to 3 standard deviations
    X = np.maximum(np.minimum(X, sigma), -sigma)/sigma
    #print X[0]
    X  = (X+1)*0.5
    #X = (X+1)*0.4 + 0.1
    #print X[0]

    grid_dim = int(np.sqrt(samp_size))
    padding = 1
    w, h = grid_dim*dim + padding*(grid_dim+1), grid_dim*dim + padding*(grid_dim+1)
    row, col = -1, -1
    grid = np.zeros((h, w))
    #grid -= 1

    for x in X:
        #print x
        #print '....'
        col += 1
        if col % grid_dim == 0:
            col = 0
            row += 1

        x_left = dim*col + (col+1)*padding
        x_right = dim*(col+1) + (col+1)*padding
        y_top = dim*row + (row+1)*padding
        y_bottom = dim*(row+1) + (row+1)*padding

        grid[y_top:y_bottom, x_left:x_right] = x.reshape(dim, dim)

    """exit()
    for row in grid:
        for cell in row:
            if cell > 0.0:
                print '@',
            else:
                print ' ',
        print ''
    exit()"""

    #plt.imshow(grid, cmap='gray', norm=Normalize(vmin=-1.0, vmax=1.0))
    plt.imshow(grid, cmap='gray', interpolation='none')
    #plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.show()

def display_decoding(x_orig, x_recon):
    # sigma = np.std(x_recon)
    # mu = np.mean(x_recon)
    # x_recon = (x_recon-mu)/sigma

    print np.linalg.norm(x_orig), np.linalg.norm(x_recon)

    dim = int(np.sqrt(len(x_orig)))
    x_orig = x_orig.reshape(dim, dim)
    x_recon = x_recon.reshape(dim, dim)

    padding = 1
    x = np.zeros((dim*2+padding*3, dim+padding*2)).T
    #x -= 1
    x[padding:padding+dim, padding:padding+dim] = x_orig
    x[padding:padding+dim, padding*2+dim:padding*2+dim*2] = x_recon

    #plt.imshow(x, cmap='gray', norm=Normalize(vmin=-1.0, vmax=1.0))
    plt.imshow(x, cmap='gray')
    #plt.imshow(x, cmap='gray', interpolation='nearest')
    plt.show()
