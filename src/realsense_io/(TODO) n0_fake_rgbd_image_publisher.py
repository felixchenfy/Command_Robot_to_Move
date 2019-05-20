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
    from utils.lib_cloud import read_color_depth_images
    from utils.lib_ros_topic import CameraInfoPublisher, ImagePublisher

# ==================================================================================================


if __name__=="__main__":
    rospy.init_node("detect_plane")
    rospy.sleep(0.5)
    bridge = CvBridge()

    # -- Names
    if 0:
        TOPIC_CAMERA_INFO = rospy.get_param("topic_camera_info")
        TOPIC_COLOR_IMAGE = rospy.get_param("topic_color_image")
        TOPIC_DEPTH_IMAGE = rospy.get_param("topic_depth_image")
        FILNAME_CAMERA_INFO = rospy.get_param("filename_camera_info")
    else:
        TOPIC_CAMERA_INFO = "/camera/color/camera_info"
        TOPIC_COLOR_IMAGE = "/camera/color/image_raw" 
        TOPIC_DEPTH_IMAGE = "/camera/aligned_depth_to_color/image_raw"
        FILNAME_CAMERA_INFO = ROOT + "config/cam_params_realsense.json"

    # -- Get color and depth image
    images_folder = ROOT + "data/raw_rgbd_images/"
    ith_image=1
    color_img, depth_img = read_color_depth_images(images_folder, ith_image,
        output_img_format="cv", index_len=5,
        image_folder="./", image_name="image_",
        depth_doler="./", depth_name="depth_",
    )

    # -- Get camera info
    camera_intrinsic = open3d.io.read_pinhole_camera_intrinsic(FILNAME_CAMERA_INFO)

    # -- Set publisher
    color_pub = ImagePublisher(TOPIC_COLOR_IMAGE, "color")
    depth_pub = ImagePublisher(TOPIC_DEPTH_IMAGE, "depth")
    cam_info_pub = CameraInfoPublisher(TOPIC_CAMERA_INFO)
    
    # Publish image
    cnt=0
    while not rospy.is_shutdown():
        cnt+=1

        color_pub.publish(color_img)
        depth_pub.publish(depth_img)
        cam_info_pub.publish_open3d_format(camera_intrinsic)

        print("node 1: pub data {:04d}...".format(cnt))
        rospy.sleep(3.0)
    print "Node0 stops"

    