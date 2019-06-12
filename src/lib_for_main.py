#!/usr/bin/env python


import numpy as np 
import tf
import cv2
import math
import rospy
from tf.transformations import rotation_matrix
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf.transformations import euler_matrix, euler_from_matrix, quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import Pose, Point, Quaternion

if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    sys.path.append(ROOT)
    
    from utils.lib_geo_trans_ros import pose2T, form_T
    
class CoodRelations(object):
    ''' Relationship (fixed transforms) between main coordinates '''
    
    @staticmethod
    def realsense_camera_link_to_color_image():
        ''' 
        WARNING: 
            Except in this function, all the other functions 
            use the word "camera" to indicate the "color_image" frame.
        '''
        if 1:
            '''
            Realsense "camera frame" to "color frame":
            $ roslaunch simon_says start_realsense_camera.launch
            $ rosrun tf tf_echo camera_link camera_color_optical_frame
            Result:
            - Translation: [-0.000, 0.015, 0.000]
            - Rotation: in Quaternion [0.504, -0.496, 0.500, -0.500]
                    in RPY (radian) [-1.572, -0.008, -1.563]
                    in RPY (degree) [-90.083, -0.451, -89.528]
            '''
            # pos = [-0.000, 0.015, 0.000]
            pos = [-0.000, 0.038, 0.000]
            quaternion = [0.504, -0.496, 0.500, -0.500]
            T = pose2T(pos, quaternion)
        else: # or may be the image is already in the camera link frame? 
            T = np.array([
                [0,  0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0,  0, 0, 1],
            ], dtype=np.float32)
        return T 
    
    @staticmethod
    def turtlebot_to_camera():
        x_dis = 0.082 # 0.055 ~ 0.1. TODO: Determine this value
        y_dis = 0 # camera is placed right in front of the turtlebot
        z_dis = 0.01 # 0.01 ~ 0.02. This value doesn't matter at all. In my case, I don't care the height
        pos = [x_dis, y_dis, z_dis]
        R = np.identity(3)
        T_bot_to_cam_link = form_T(R, pos) 
        T_cam_link_to_color = CoodRelations.realsense_camera_link_to_color_image()
        
        # return 
        T_bot_to_color = np.dot(T_bot_to_cam_link, T_cam_link_to_color)
        T_bot_to_camera = T_bot_to_color # this is just a rename
        return T_bot_to_camera 

# class CoodTrans(object):
#     ''' Transform a point'
#     pass         

if __name__=="__main__":
    def main():
        T = CoodRelations.turtlebot_to_camera()
        print(T) 

if __name__=="__main__":
    node_name = 'library_for_main_script'
    rospy.init_node(node_name)
    rospy.sleep(0.2)
    main()
    