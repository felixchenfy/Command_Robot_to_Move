#!/usr/bin/env python
''' 
Testing the Detector Class.
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
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    sys.path.append(ROOT)
    
    import utils.lib_geo_trans as lib_geo_trans 
    import utils.lib_commons as lib_commons 
    
    from config.config import set_args
    
    # main classes for performing plane and object detection
    from detection.detect_plane import PlaneDetector 
    from detection.yolo_request import FakeYoloDetector

def cv2_show(img_disp, ith_img, wait_key_ms):
    ''' 
    Put a number at the top-left of img_disp, and then cv2.imshow().
    WARNING: This changes img_disp !!!
    '''
    cv2.putText(img_disp, str(ith_img), (50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=2, color=(0, 0, 0), thickness=2)

    cv2.imshow("", img_disp)
    q = cv2.waitKey(wait_key_ms)
    
    good_key = not (q>0 and chr(q)=='q')
    return good_key

def find_target(
        labels_and_pos, # detected objects. A list of (label, pos)
        TARGET_OBJECT, 
        depth_img, 
        intrinsic_matrix,
    ):
    p_target_2d, p_target_3d = None, None 
    
    for label, pos in labels_and_pos: # get p_target_2d
        if label == TARGET_OBJECT:
            p_target_2d = pos 
            break 
    
    if p_target_2d is not None: # get p_target_3d
        v, u = p_target_2d
        depth = depth_img[u][v] / 1000.0
        P = np.array([v, u, 1])
        K = intrinsic_matrix
        p_target_3d =  depth * np.dot(np.linalg.inv(K), P)
    return p_target_2d, p_target_3d

import datetime

def save_images(color_img, depth_img, result_img):
    s=str(datetime.datetime.now())[5:].replace(' ','-').replace(":",'-').replace('.','-')[:-3]
    s_base = ROOT + "/tmp/" 
    imgs = [color_img, depth_img, result_img]
    keys = ["color", "depth", "result"]
    for i in range(3):
        folder = s_base + keys[i] + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(folder + s + "_color.jpg", imgs[i])
    
def main():
    
    classes = ["one", "two", "three", "four", "five"]
    TARGET_OBJECT = "three"
    assert TARGET_OBJECT in classes

    plane_detector = PlaneDetector()
    fake_yolo_detector = FakeYoloDetector(classes=classes)
    
    ith_img = 0
    while not rospy.is_shutdown():
        print("="*80)
        print("TARGET OBJECT = {}".format(TARGET_OBJECT))
        
        # Get image
        ith_img += 1
        color_img, depth_img, cloud = plane_detector.sub_rgbd_and_cloud()
        img_disp = color_img.copy()
        
        # Detect plane and object
        p_cent, v_norm, p_target_2d, p_target_3d = [None]*4 # init output
        
        if True: # Detect plane, its center and norm
            
            p_cent, v_norm, cloud_plane = plane_detector.detect_plane(
                cloud, min_points=2000)
            
            if p_cent is None:
                print("Plane not detected -- plane points are too few.")
            else: # draw
                img_disp = plane_detector.draw_plane_onto_color_image(img_disp, cloud_plane)
                img_disp = plane_detector.draw_plane_norm(img_disp, p_cent, v_norm)
            
        if True: # Detect and locate the target object
            
            # Detect objects, get a list of (label, pos)
            detections = fake_yolo_detector.detect(color_img, if_print=False)
            labels_and_pos  = fake_yolo_detector.detetions_to_labels_and_pos(detections)
                   
            # Find the object's pos: p_target_2d, p_target_3d
            p_target_2d, p_target_3d = find_target(
                labels_and_pos, TARGET_OBJECT, depth_img, plane_detector.intrinsic_matrix)
            
            if 1:
                for label, pos in labels_and_pos:
                    print("Detect '{}', pos = {}".format(label, pos))

                img_disp = fake_yolo_detector.draw_detections_onto_image( # draw bbox
                    img_disp, detections, classes)
            
                if p_target_3d is not None: # draw target object
                    img_disp = plane_detector.draw_plane_norm(
                        img_disp, p_target_3d, v_norm, color=[0, 255, 0])

        
        # display
        print("p_cent = {}".format(p_cent))    
        print("v_norm = {}".format(v_norm))    
        print("p_target_2d = {}".format(p_target_2d))    
        print("p_target_3d = {}".format(p_target_3d))    
        print("="*80)
        
        if not cv2_show(img_disp, ith_img, wait_key_ms=50):
            break 

        # save
        save_images(color_img, depth_img, img_disp)
        
        rospy.sleep(1.0)
    
if __name__ == '__main__':
    node_name = 'detect_plane_from_point_cloud'
    rospy.init_node(node_name)
    rospy.sleep(1)
    main()
    cv2.destroyAllWindows()
    