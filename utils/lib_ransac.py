
# Copy and modify from here
# https://scipy-cookbook.readthedocs.io/items/RANSAC.html 

import numpy as np
import scipy # use np if scipy unavailable
import scipy.linalg # use np if scipy unavailable
import time

def ransac(
    data, # P*N: p points, N feature dimension
    model,  # A class with methods: 
            #   weight = fit(data)
            #   error = get_error(data, weights)
    n_pts_base, # number of points to fit a basic model
    n_pts_extra, # min number of extra points for a good model
    max_iter,
    dist_thre,
    print_time=False,
    debug=False):

    P, N = data.shape[0], data.shape[1]
    assert n_pts_base < P
    print("\n\n--------------------------------")
    print("Start RANSAC algorithm ...")
    print("Input: num points = {}, features dim = {}".format(P, N))
    print("Config: n_pts_base = {}, n_pts_extra = {}, dist_thre = {}".format(
        n_pts_base, n_pts_extra, dist_thre))
    print("")

    t0 = time.time()
    iter = 0
    best_weights = None
    best_err = np.inf
    best_inlier_idxs = None
    for iter in range(max_iter):
        


        # Get may_be data, and fit may_be model
        maybe_idxs, test_idxs = random_partition(n_pts_base, P)
        maybe_inliers = data[maybe_idxs,:]
        maybe_weights = model.fit(maybe_inliers)

        # Evaluate on test data
        test_points = data[test_idxs]
        test_err = model.get_error(test_points, maybe_weights)
        also_idxs = test_idxs[test_err < dist_thre] # select indices of rows with accepted points
        also_inliers = data[also_idxs, :]

        if debug and (iter == max_iter-1 or iter % 1 == 0):
            print("\n{}th iteration".format(iter))
            print('\ttest_err: mean = {}, min = {}, max = {}'.format(
                test_err.mean(), test_err.min(), test_err.max()))
            print('\ttotal fit points = {}'.format(len(also_inliers)))
            
        # Fit again
        if len(also_inliers) > n_pts_extra:
            better_data = np.concatenate( (maybe_inliers, also_inliers) )
            better_weights = model.fit(better_data)
            better_errors = model.get_error(better_data, better_weights)
            curr_error = np.mean( better_errors )
            if curr_error < best_err:
                best_weights, best_err = better_weights, curr_error
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
        iter+=1

        # Output
    if best_weights is None:
        raise ValueError("Didn't find any good model")

    # Print time
    if print_time:
        print("Time cost for RANSAC = {:.3f} seconds".format(time.time() - t0))
    print("--------------------------------\n")
    return best_weights, best_inlier_idxs

def random_partition(n, N):
    # get n random indices out of [0, 1, 2, ..., N]
    indices = np.random.permutation(N)
    return indices[:n], indices[n:]
