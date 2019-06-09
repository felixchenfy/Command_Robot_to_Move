#!/usr/bin/env python
''' 
Testing the PlaneDetector Class.
This script runs in ROS.
'''

if 1:  # common
    import argparse
    import numpy as np
    import sys
    from time import time
    import math
    import open3d
    import cv2 
    
if 1:  # ros
    import rospy
    import roslib
    import rospy

if 1:  # ros geo & tf
    import tf
    from std_msgs.msg import Header
    from geometry_msgs.msg import Point, Quaternion, Pose, Twist, Vector3
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState, ModelStates
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Empty

if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    
    import utils.lib_geo_trans as lib_geo_trans 
    import utils.lib_commons as lib_commons 
    
    from config.config import set_args
    from utils.lib_ros_topic import ColorImageSubscriber, DepthImageSubscriber
    from utils.lib_cloud import read_color_and_depth_image, rgbd2cloud, getCloudContents, formNewCloud, filtCloudByRange
    from utils.lib_ransac import ransac
    from utils.lib_plane import PlaneModel
    from utils.lib_plot import show, plot_cloud_3d
    
def take_phote():
    pass 

class PlaneDetector(object):
    def __init__(self, IF_ROS=True):        
        self.args = set_args()
        
        # init vars in ros
        if IF_ROS:
            self._init_vars_in_ros()
        
        # load camera info
        self.camera_intrinsics = open3d.io.read_pinhole_camera_intrinsic(
            self.args["filename_camera_info"]) # camera
        self.intrinsic_matrix = self.camera_intrinsics.intrinsic_matrix

        # save other vars
        pass 
    
    def _init_vars_in_ros(self):
        
        # image subscriber
        self.sub_color = ColorImageSubscriber(self.args["topic_color_image"])
        self.sub_depth = DepthImageSubscriber(self.args["topic_depth_image"])
        
    def sub_rgbd_and_cloud(self, 
                           compute_point_cloud=True,
                           voxel_size=0.005):
        sub_color, sub_depth = self.sub_color, self.sub_depth
        while not rospy.is_shutdown():
            if(sub_color.isReceiveImage() and sub_depth.isReceiveImage):
                color_img, t1 = sub_color.get_image()
                depth_img, t2 = sub_depth.get_image()
                break
            rospy.sleep(0.05)
        
        if compute_point_cloud:
            cloud = self.compute_point_cloud(color_img, depth_img)
            cloud = open3d.geometry.voxel_down_sample(cloud, voxel_size=voxel_size)
        return color_img, depth_img, cloud
    
    def compute_point_cloud(self, color_img, depth_img):
        rgbd_image = open3d.create_rgbd_image_from_color_and_depth(
            open3d.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), 
            open3d.Image(depth_img),
            convert_rgb_to_intensity=False)
        cloud = open3d.create_point_cloud_from_rgbd_image(
            rgbd_image, self.camera_intrinsics)
        return cloud 
    
    def detect_plane(self, cloud):
        ''' Use RANSAC to detect plane from point cloud. '''
        '''
        The cloud will be range-filtered by prior knowledge before RANSAC. 
        This function can be tested in the notebook "test_detect_plane.ipynb".
        '''
        
        cloud = filtCloudByRange(cloud, zmax=1.0, ymax=0.1)
        
        # Use RANSAC to detect plane
        xyz, color = getCloudContents(cloud) # color: rgb
        model = PlaneModel(feature_dimension=3)
        coefs, inliers = ransac(
            xyz,
            model, 
            n_pts_base=3,
            n_pts_extra=30,
            max_iter=100,
            dist_thre=0.01,
            print_time=True,
            debug=False,
        )
        print("source points = {}, inliers points = {}".format(
            xyz.shape[0], inliers.size ))

        # Get inliers points
        plane_xyz, plane_color = xyz[inliers, :], color[inliers, :]
        cloud_plane = formNewCloud(plane_xyz, plane_color)
        
        # Downsample
        if 1:
            cloud_plane = open3d.geometry.voxel_down_sample(
                cloud_plane, voxel_size=0.02)
            plane_xyz, plane_color = getCloudContents(cloud_plane)
        
        # Get plane center
        if 0: # use median
            p_cent = np.median(plane_xyz, axis=0) 
        else: # remove large and small points, and then take the average
            p_cent = np.zeros((3, ))
            P = plane_xyz.shape[0] # number of points
            for i, dim_i in enumerate(plane_xyz.T):
                dim_i = np.sort(dim_i)
                n1 = int(round(P*0.3 - 0.5))
                n2 = int(round(P*0.7 + 0.5))
                p_cent[i] = np.mean(dim_i[n1:n2+1])
                
        # Get plane norm
        v_norm = coefs[1:]
        if v_norm[2] > 0:
            v_norm = -v_norm
        
        return p_cent, v_norm, cloud_plane
            

    def draw_plane_onto_color_image(
            self, 
            img, 
            cloud_plane, # open3d cloud
        ):
        
        # Check input
        camera_intrinsics = self.camera_intrinsics
        
        # Project point from 3d to image 
        xyz, color = getCloudContents(cloud_plane)
        mask, vu_s = lib_geo_trans.project_points_to_image_mask(
            xyz, self.camera_intrinsics, scale=5.0,
        )
        
        # Draw image
        color_offset = [0, 0, 150] # b,g,r
        img_disp = lib_commons.increase_color(
            img, mask, color_offset)
        
        return img_disp
    
    
    def draw_plane_norm(
            self, 
            img_disp, 
            p_on_plane, 
            vec_norm,
            color=[0,0,255], 
            len_arrow = 0.3, # meter
        ):
        
        # Check input
        intrinsic_matrix = self.intrinsic_matrix
        def to_int_tuple(arr):
            return (int(arr[0]), int(arr[1]))
        
        # Draw plane center
        p0_xy = lib_geo_trans.cam2pixel(
            p_on_plane.reshape((-1, 1)), intrinsic_matrix)
        p0_xy = to_int_tuple(p0_xy)
        cv2.circle(img_disp, p0_xy, radius=5, color=color, 
                thickness=5, lineType=cv2.LINE_AA)

        # Draw plane norm
        p1 = p_on_plane + len_arrow * vec_norm
        p1_xy = lib_geo_trans.cam2pixel(
            p1.reshape((-1, 1)), intrinsic_matrix)
        p1_xy = to_int_tuple(p1_xy)
        cv2.arrowedLine(img_disp, p0_xy, p1_xy, color=color, 
                        thickness=5, tipLength=0.3)
        
        return img_disp

# class PointPlanner(object):
#     ''' Based on object position, plan the point to go for the robot '''
    
#     @ staticmethod
#     def compute_P2_point_in_front_of_plane(
#             p_cent, 
#             v_norm,
#             l_view=1.0, # the distance between robot and plane for taking a picture
#         ):
#         p_view = p_cent + l_view * v_norm
        
#         # transform p_view to the robot coordinate
#         # TODO
        
#         return p_view 
    
def main():
    
    plane_detector = PlaneDetector()
    
    while not rospy.is_shutdown():
        
        color_img, depth_img, cloud = plane_detector.sub_rgbd_and_cloud()
        img_disp = color_img.copy()
        
        # detect
        p_cent, v_norm, cloud_plane = plane_detector.detect_plane(cloud)

        # draw result
        img_disp = plane_detector.draw_plane_onto_color_image(img_disp, cloud_plane)
        img_disp = plane_detector.draw_plane_norm(img_disp, p_cent, v_norm)
        
        # display
        cv2.imshow("", img_disp)
        q = cv2.waitKey(50)                
        if q>0 and chr(q)=='q':
            break 
        
        print("p_cent = {}".format(p_cent))    
        print("v_norm = {}".format(v_norm))    
        print("="*80)
        rospy.sleep(1.0)
    
if __name__ == '__main__':
    node_name = 'detect_plane_from_point_cloud'
    rospy.init_node(node_name)
    rospy.sleep(1)
    main()
    cv2.destroyAllWindows()
    