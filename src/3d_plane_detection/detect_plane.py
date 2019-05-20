#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

if 1: # common
    import numpy as np
    import copy
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # THIS MUST BE IMPORTED!
    import open3d
    import time 
    import pcl

if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    from utils.lib_cloud import read_color_and_depth_image, rgbd2cloud, getCloudContents, formNewCloud, filtCloudByRange
    from utils.lib_ransac import ransac
    from utils.lib_plane import fit_plane_by_PCA, abc_to_w, create_plane

# ==================================================================================================

class PlaneModel(object):

    def __init__(self, feature_dimension=3):
        self.feature_dimension = feature_dimension
    
    def fit(self, data): # Fit plane by PCA
        # data: P*N
        # w:    N
        w = self.fit_plane_by_PCA(data, N=self.feature_dimension)
        return w 

    def get_error(self, data, w):
        point_to_plane_distance = (w[0] + np.dot(data, w[1:])) / np.linalg.norm(w[1:])
        return np.abs(point_to_plane_distance)
    
    def fit_plane_by_PCA(self, X, N=3):
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


if __name__=="__main__":
    # -- Get color and depth image
    images_folder = ROOT + "data/plane_detection/"
    ith_image=1
    color_img, depth_img = read_color_and_depth_image(images_folder, ith_image,
        output_img_format="open3d", index_len=1,
        image_folder="./", image_name="image_",
        depth_doler="./", depth_name="depth_",
    )

    # -- Get camera info
    FILNAME_CAMERA_INFO = ROOT + "config/cam_params_realsense.json"
    camera_intrinsic = open3d.io.read_pinhole_camera_intrinsic(FILNAME_CAMERA_INFO)


    # -- Get point cloud
    cloud = rgbd2cloud(color_img, depth_img, camera_intrinsic, input_img_format="open3d")
    cloud = open3d.geometry.voxel_down_sample(cloud, voxel_size=0.02)
    cloud = filtCloudByRange(cloud, zmax=0.8)
    xyz, color = getCloudContents(cloud)

    # Draw whole cloud
    if 1:
        open3d.draw_geometries([cloud])

    # segment plane
    if 0:
        pcl_cloud = pcl.PointCloud(xyz.astype(np.float32))
        seg = pcl_cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        inliers, model = seg.segment()
    else:
        # Call ransac
        model = PlaneModel(feature_dimension=3)
        w, inliers = ransac(
            xyz,
            model, 
            n_pts_base=20,
            n_pts_extra=40,
            max_iter=10,
            dist_thre=0.01,
            print_time=True,
            debug=False,
        )
    print("source points = {}, inliers points = {}".format(xyz.shape[0], inliers.size ))
    
    # Get inlier points
    plane_xyz = xyz[inliers, :]
    plane_color = color[inliers, :]
    plane_open3d_cloud = formNewCloud(plane_xyz, plane_color)

    # Draw plane by Open3D
    if 1:
        open3d.draw_geometries([plane_open3d_cloud])
 
    # Draw plane by plt
    if 0:
        # -- Plot source points
        fig = plt.figure().gca(projection='3d')
        # fig.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
        fig.scatter(xs=plane_xyz[:, 0], ys=plane_xyz[:, 1], zs=plane_xyz[:, 2])

        # -- Plot fitted plane
        xx, yy, zz = create_plane(
            weights_w=w, 
            point_gap=0.2, 
            xy_range=(-1, 1, -1, 1), 
            noise=0,
            )
        fig.plot_surface(X=xx, Y=yy, Z=zz, alpha=1.0, rstride=1, cstride=1, cmap='rainbow')

        # -- Final show
        plt.show()

