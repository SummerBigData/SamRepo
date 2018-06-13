import matplotlib.pyplot as plt
import numpy as np

def display_sample(X, samp_size=100):
    print X.shape
    X = (X + 1.0) / 2.0
    dim = int(np.sqrt(X.shape[1]/3.0))

    idxs = list(range(len(X)))
    idxs = np.random.choice(idxs, samp_size, replace=False)
    samp = X[idxs]

    grid_dim = int(np.sqrt(samp_size))
    padding = 2
    w, h = grid_dim*dim + padding*(grid_dim+1), grid_dim*dim + padding*(grid_dim+1)
    row, col = -1, -1
    grid = np.zeros((h, w, 3))
    grid -= 1

    for s in samp:
        col += 1
        if col % grid_dim == 0:
            col = 0
            row += 1
    
        x_left = dim*col + (col+1)*padding
        x_right = dim*(col+1) + (col+1)*padding
        y_top = dim*row + (row+1)*padding
        y_bottom = dim*(row+1) + (row+1)*padding
        
        grid[y_top:y_bottom, x_left:x_right] = s.reshape((dim, dim, 3))

    plt.imshow(grid)
    plt.show()

def display_encoding(W, samp_size=400):
    W = (W + 1.0) / 2.0
    dim = int(np.sqrt(W.shape[1]/3.0))
    print W.shape, dim

    grid_dim = int(np.sqrt(samp_size))
    padding = 2
    w, h = grid_dim*dim + padding*(grid_dim+1), grid_dim*dim + padding*(grid_dim+1)
    row, col = -1, -1
    grid = np.zeros((h, w, 3))
    #grid -= 1

    for x in W:
        col += 1
        if col % grid_dim == 0:
            col = 0
            row += 1

        x_left = dim*col + (col+1)*padding
        x_right = dim*(col+1) + (col+1)*padding
        y_top = dim*row + (row+1)*padding
        y_bottom = dim*(row+1) + (row+1)*padding

        s = dim**2
        img = np.zeros((dim, dim, 3))
        img[:,:,0] = x[:s].reshape(dim, dim)
        img[:,:,1] = x[s:2*s].reshape(dim, dim)
        img[:,:,2] = x[2*s:].reshape(dim, dim)

        grid[y_top:y_bottom, x_left:x_right] = img

    plt.imshow(grid, interpolation='nearest')
    plt.show()

def display_decoding(x_orig, x_recon):
    print np.linalg.norm(x_orig), np.linalg.norm(x_recon)

    dim = int(np.sqrt(len(x_orig)))
    x_orig = x_orig.reshape(dim, dim)
    x_recon = x_recon.reshape(dim, dim)

    padding = 1
    x = np.zeros((dim*2+padding*3, dim+padding*2)).T
    x -= 1
    x[padding:padding+dim, padding:padding+dim] = x_orig
    x[padding:padding+dim, padding*2+dim:padding*2+dim*2] = x_recon

    plt.imshow(x)
    plt.show()