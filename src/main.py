#!/usr/bin/env python

'''
currently this is only a pseudo-code
TODO: complete this
'''

if 1:  # common
    import argparse
    import numpy as np
    import sys
    from time import time
    import math

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

def take_phote():
    pass 

if __name__=="__main__":
    
    target_label = set_target_through_voice()
    
    # -- Phase 1: Move robot in the front of the plane
    color_img, depth_img = take_phone()
    plane_center, plane_direction = detect_plane(depth_img)
    P2 = compute_P2_point_in_front_of_plane(
        plane_center, plane_direction
    )
    move_robot_to(P2)
    
    # -- Phase 2: Locate target and go there
    color_img, depth_img = take_phone()
    P_target = detect_object(color_img, depth_img, target_label)
    plane_center, plane_direction = detect_plane(depth_img)
    P3 = compute_P3_point_near_plane(
        P_target, plane_direction
    )
    move_robot_to(P3)
    move_robot_forward()
    
    # -- Complete
    