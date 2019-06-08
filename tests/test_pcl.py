#!/usr/bin/env python2
'''
WARNING: This script is deprecated
'''

from __future__ import division
import pcl
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # THIS MUST BE IMPORTED!
 
if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    sys.path.append(ROOT)
    from utils.lib_cloud import read_color_and_depth_image, rgbd2cloud, getCloudContents, formNewCloud, filtCloudByRange
    from utils.lib_ransac import ransac
    from utils.lib_plane import fit_plane_by_PCA, abc_to_w, create_plane

# cloud = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
cloud = np.random.random((10, 3), dtype=np.float32)
p = pcl.PointCloud(cloud)
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
indices, model = seg.segment()

