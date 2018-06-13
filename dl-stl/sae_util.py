import numpy as np
import scipy.optimize as opt
from scipy.special import expit
from sklearn.preprocessing import normalize

def generate_sample(imgs, samp_size=10000):
    patch_size = 8
    dim = int(np.sqrt(imgs.shape[1]))
    X = np.zeros((samp_size, patch_size**2))

    for i in range(samp_size):
        idx = np.random.randint(imgs.shape[0])
        img = imgs[idx].reshape(dim, dim)
        base = np.random.randint(dim-patch_size)
        X[i] = img[base:base+patch_size, base:base+patch_size].flatten()
        
    X -= np.mean(X)
    sigma = 3*np.std(X) # truncate to 3 standard deviations
    X = np.maximum(np.minimum(X, sigma), -sigma)/sigma # truncate and scale to [-1, 1]
    X  = (X+1)*0.5
    return X

def forward_prop(X, Ws, bs):
    m = X.shape[0]
    a = X
    a_arr = []

    for W, b in zip(Ws, bs):
        #print a.shape, W.shape, b.shape
        a_arr.append(a)
        z = a.dot(W.T)
        a = expit(z + np.tile(b.flatten(), (m, 1)))

    a_arr.append(a)
    return a_arr, np.mean(a_arr[-2], axis=0)

def cost(h, y, m, lmbda=0, Ws=[], beta=0, rho=0.05, rhs=np.array([])):
    cost = np.mean(np.sum((h - y)**2, axis=1)) / 2.0

    if lmbda > 0:
        #weight_sum = sum(list(map(
        #    lambda W: np.sum(W.flatten()**2), Ws)))
        weight_sum = 0
        for W in Ws:
            weight_sum += np.sum(W.flatten()**2)
        cost += weight_sum * lmbda / 2.0

    if beta > 0:
        kl_cost = np.sum(
            rho * np.log(rho / rhs) + 
            (1-rho) * np.log((1-rho) / (1-rhs)))
        cost += beta * kl_cost

    return cost

def fprime(a):
    return np.multiply(a, 1-a)

global_step = 0
def back_prop(a_arr, rhs, y, m, L, Ws, bs, lmbda=0, rho=0.05, beta=0):
    global global_step
    global_step += 1
    if global_step % 20 == 0:
        print 'Global Step: %d' % (global_step)

    W1, W2 = Ws[0], Ws[1] # W1 - 3x64, W2 - 64x3
    b1, b2 = bs[0], bs[1] # b1 - 3x1, b2 - 64x1
    a1, a2, a3 = a_arr[0], a_arr[1], a_arr[2]

    dout = np.multiply((a3 - y), fprime(a3)) # 100x64

    rho_vec = beta * (-rho/rhs + (1-rho)/(1-rhs)) # 3,
    rho_vec = rho_vec.reshape((1, len(rho_vec)))
    rho_vec = np.tile(rho_vec, (m, 1)) # 100x3

    dhidden = np.multiply(dout.dot(W2) + rho_vec, fprime(a2)) # (100x64 . 64x3) = 100x3

    dW2 = dout.T.dot(a2)/float(m) # (64x100 . 100x3) = 64x3
    dW1 = dhidden.T.dot(a1)/float(m) # (3x100 . 100x64) = 3x64
    db2 = np.mean(dout, axis=0).reshape(b2.T.shape).T
    db1 = np.mean(dhidden, axis=0).reshape(b1.T.shape).T

    if lmbda > 0:
        dW1 += lmbda * W1
        dW2 += lmbda * W2

    g = [dW1, db1, dW2, db2]
    #g = list(map(lambda d: d/float(m), g))
    return g

def random_weights(sizes):
    def eps(size):
        return np.sqrt(6.0/(size[0]+size[1]+1.0))
    return [np.random.uniform(-eps(s), eps(s), s) for s in sizes]

def weights_from_flat(theta_flat, sizes):
    Ws, bs = [], []
    idx = 0
    for i, size in enumerate(sizes):
        n = size[0]*size[1]
        if i % 2 == 0:
            Ws.append(
                np.array(theta_flat[idx:idx+n]).reshape(size))
        else:
            bs.append(
                np.array(theta_flat[idx:idx+n]).reshape(size))
        idx += n
    return Ws, bs

def check_grad(X, y, m, n, L, sizes, lmbda=0, rho=0.05, beta=0):
    Ws = random_weights(sizes)
    bs = random_weights([(size[0], 1) for size in sizes])

    theta_flat = np.array([], dtype=float)
    for W, b in zip(Ws, bs):
        theta_flat = np.append(theta_flat, W.flatten())
        theta_flat = np.append(theta_flat, b.flatten())

    tmp_sizes = []
    for size in sizes:
        tmp_sizes.append(size)
        tmp_sizes.append((size[0], 1))
    sizes = tmp_sizes[:]

    def f(thetas, *args):
        Ws, bs = weights_from_flat(thetas, sizes)
        h, rhs = forward_prop(X, Ws, bs)
        return cost(h[-1], y, m, lmbda=lmbda, Ws=Ws, beta=beta, rho=rho, rhs=rhs)
    
    def fgrad(thetas, *args):
        Ws, bs = weights_from_flat(thetas, sizes)
        a_arr, rhs = forward_prop(X, Ws, bs)	
        dels = back_prop(a_arr, rhs, y, m, L, Ws, bs, lmbda=lmbda, rho=rho, beta=beta)
        g = np.array([], dtype=float)
        for d in dels:
            g = np.append(g, d.flatten())
        return g

    return opt.check_grad(f, fgrad, theta_flat)

def train(X, m, n, L, sizes, lmbda=0, rho=0.05, beta=0, niter=400):
    Ws = random_weights(sizes)
    bs = random_weights([(size[0], 1) for size in sizes])

    theta_flat = np.array([], dtype=float)
    for W, b in zip(Ws, bs):
        theta_flat = np.append(theta_flat, W.flatten())
        theta_flat = np.append(theta_flat, b.flatten())

    tmp_sizes = []
    for size in sizes:
        tmp_sizes.append(size)
        tmp_sizes.append((size[0], 1))
    sizes = tmp_sizes[:]

    def f(thetas, *args):
        Ws, bs = weights_from_flat(thetas, sizes)
        h, rhs = forward_prop(X, Ws, bs)
        J = cost(h[-1], X, m, lmbda=lmbda, Ws=Ws, beta=beta, rho=rho, rhs=rhs)
        
        if global_step % 20 == 0:
            print 'Cost: ', J
            
        return J
    
    def fgrad(thetas, *args):
        Ws, bs = weights_from_flat(thetas, sizes)
        a_arr, rhs = forward_prop(X, Ws, bs)	
        dels = back_prop(a_arr, rhs, X, m, L, Ws, bs, lmbda=lmbda, rho=rho, beta=beta)

        g = np.array([], dtype=float)
        for d in dels:
            g = np.append(g, d.flatten())
        #print g
        return g

    return opt.minimize(
        f, theta_flat, jac=fgrad, 
        method='CG',
        options={'disp': True, 'maxiter': niter}).x
