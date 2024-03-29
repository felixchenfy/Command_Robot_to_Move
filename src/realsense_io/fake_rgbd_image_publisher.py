#!/usr/bin/env python
# -*- coding: utf-8 -*-

if 1: # common
    import numpy as np
    import copy
    import cv2
    from matplotlib import pyplot as plt
    import open3d
    from open3d import *
    import time
    import glob
if 1: # ROS
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge, CvBridgeError
    from std_msgs.msg import String

if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    from utils.lib_cloud import read_color_and_depth_image
    from utils.lib_ros_topic import CameraInfoPublisher, ImagePublisher
    from config.config import set_args

# ==================================================================================================


if __name__=="__main__":
    rospy.init_node("detect_plane")
    rospy.sleep(0.5)
    bridge = CvBridge()

    # -------------------- Setings --------------------
    args = set_args()

    # images folder
    images_folder = ROOT + "data/test_images/" 
    TEST_ALL = 1
    if not TEST_ALL:
        IMAGE_IDX_TO_TEST = 11
        # IMAGE_IDX_TO_TEST < 17, since there are 17 images 
    fnames = sorted(glob.glob(images_folder + "image_*.png"))
    
    # -- Topic names
    if 0:
        TOPIC_CAMERA_INFO = rospy.get_param("topic_camera_info")
        TOPIC_COLOR_IMAGE = rospy.get_param("topic_color_image")
        TOPIC_DEPTH_IMAGE = rospy.get_param("topic_depth_image")
        FILNAME_CAMERA_INFO = rospy.get_param("filename_camera_info")
    else:
        TOPIC_CAMERA_INFO = args["topic_camera_info"]
        TOPIC_COLOR_IMAGE = args["topic_color_image"]
        TOPIC_DEPTH_IMAGE = args["topic_depth_image"]
        FILNAME_CAMERA_INFO = args["filename_camera_info"] 
    
    # -- Get camera info
    camera_intrinsic = open3d.io.read_pinhole_camera_intrinsic(FILNAME_CAMERA_INFO)

    # -- Set publisher
    color_pub = ImagePublisher(TOPIC_COLOR_IMAGE, "color")
    depth_pub = ImagePublisher(TOPIC_DEPTH_IMAGE, "depth")
    cam_info_pub = CameraInfoPublisher(TOPIC_CAMERA_INFO)
    
    # -- Publish image
    cnt=0
    idx_test = 0

    while not rospy.is_shutdown():
        cnt+=1
        
        if not TEST_ALL:
            idx_test = IMAGE_IDX_TO_TEST
            
        print(fnames[idx_test])
        s = fnames[idx_test].split('/')[-1].split('.')[0][6:]
        image_name = "image_" + s + ".png"
        depth_name = "depth_" + s + ".png" 
        
        # -- Get color and depth image
        color_image = cv2.imread(images_folder + image_name, cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(images_folder + depth_name, cv2.IMREAD_UNCHANGED)
        assert (color_image is not None), "Fail to read in image from folder:\n\t{}".format(images_folder)
    
        # -- Publish
        color_pub.publish(color_image)
        depth_pub.publish(depth_image)
        cam_info_pub.publish_open3d_format(camera_intrinsic)

        print("node 1: pub data {:04d}...".format(cnt))
        rospy.sleep(3.0)
        
        # loop
        idx_test += 1
        if idx_test == len(fnames):
            idx_test = 0
    
    print("Node0 stops")

    