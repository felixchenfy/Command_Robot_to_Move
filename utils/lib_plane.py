
import numpy as np 

def plane_model(x, y, w=None, abc=None):
    if w is not None: # w0 + w1*x + w2*y + w3*z = 0
        z = (-w[0] - w[1]*x - w[2]*y) / w[3]
    if abc is not None:
        a, b, c = abc
        z = a * x + b * y + c
    return z

def abc_to_w(abc):
    # z = ax + by + c ==>  c +  a*x +  b*y + -1*z = 0
    #                     w0 + w1*x + w2*y + w3*z = 0
    a, b, c = abc
    return [c, a, b, -1]

def w_to_abc(w):
    # z = ax + by + c ==>  c +  a*x +  b*y + -1*z = 0
    #                     w0 + w1*x + w2*y + w3*z = 0
    w0, w1, w2, w3 = w
    a, b, c = w1/(-w3), w2/(-w3), w0/(-w3)
    return [a, b, c]

def create_plane(
    weights_w=None, weights_abc=None,
    xy_range=(-5, 5, -5, 5), point_gap=1.0, noise=0,
    format="2D"):
    # For line weights, choose one of "w" or "abc"

    # Check input    
    GAP = point_gap
    xl, xu, yl, yu = xy_range
    assert xl < xu and yl < yu 

    # Create data
    xx, yy = np.mgrid[xl:xu:GAP, yl:yu:GAP]
    zz = plane_model(xx, yy, w=weights_w, abc=weights_abc)
    zz += np.random.random(zz.shape) * noise # add noise
    
    # Output
    if format == "1D":
        x, y, z = map(np.ravel, [xx, yy, zz]) # matrix to array
        return x, y, z 
    else: # 2D 
        return xx, yy, zz 

def fit_plane_by_PCA(X, N=3):
    # N: number of features

    # Check input X.shape = (P, 3)
    if X.shape[0] == N: X = X.T
    
    '''
    U, S, W = svd(Xc)
    if X=3*P, U[:, -1], last col is the plane norm
    if X=P*3, W[-1, :], last row is the plane norm
    Besides, S are the eigen values
    '''
    xm = np.mean(X, axis=0) # 3
    Xc = X - xm[np.newaxis, :]
    U, S, W = np.linalg.svd(Xc)
    plane_normal = W[-1, :] # 3
    
    '''
    Compute the bias:
    The fitted plane is this: w[1]*(x-xm)+w[2]*(x-ym)+w[3]*(x-zm)=0
    Change it back to the original:w[1]x+w[2]y+w[3]z+(-w[1]xm-w[2]ym-w[3]zm)=0
        --> w[0]=-w[1]xm-w[2]ym-w[3]zm
    '''
    w_0 = np.dot(xm, -plane_normal)
    w_1 = plane_normal
    w = np.concatenate(([w_0], w_1))
    return w 
