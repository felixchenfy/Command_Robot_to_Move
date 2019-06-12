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
    
    from detection.detect_plane import PlaneDetector 
    from detection.yolo_request import FakeYoloDetector
    from speech_recognition.voice_listener import VoiceListener
    from turtlebot_control.lib_turtlebot import Turtle
    
    from config.config import set_args
    
    # main classes for performing plane and object detection
    from lib_for_main import CoodRelations

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

class CoordManager(object):
    ''' Manage coordinate transformations for this project '''
    def __init__(self):
        self.T_bot_to_cam = CoodRelations.turtlebot_to_camera()
        
    # def p3d_cam_to_bot(self, p_in_cam):
    #     p_in_bot = np.dot(self.T_bot_to_cam, np.array(p_in_cam))
    #     return p_in_bot 
    
    def trans_pos(self, T4x4, p3):
        assert len(p3) == 3
        p4 = np.append(p3, 1)
        p4 = np.dot(T4x4, p4)
        p3 = p4[0:3]
        return p3 
    
    def get_pos_and_dir_in_robot(
            self, 
            p_in_cam, # len=3
            direction_in_cam, # len=3 
            l_offset=0, # robot needs to have a distance to the target.
        ):
        '''
        Get the target pos and direction, for robot to move there.
        '''
        
        # Check input
        if p_in_cam is None or direction_in_cam is None: 
            return None, None 
        
        if not direction_in_cam[2] < 0: # this direction should be pointing to the robot
            direction_in_cam *= -1
        # direction in camera xz plane should have a length of 1
        n_ = direction_in_cam
        direction_in_cam /= np.linalg.norm([n_[0], n_[2]]) 
        
        if l_offset:
            p_in_cam += direction_in_cam * l_offset
            
        # Change coordinate for 2 points, i.e. two ends of the arrow
        p0 = p_in_cam
        p1 = p_in_cam + 1 * direction_in_cam
        p0 = self.trans_pos(self.T_bot_to_cam, p0) # change to robot's frame
        p1 = self.trans_pos(self.T_bot_to_cam, p1) # change to robot's frame
        x0, y0 = p0[0:2] # we only care about [x, y]. No need for z
        x1, y1 = p1[0:2] 
        pos_in_bot = np.array([x0, y0])
        
        # compute desired robot direction
        direction_in_bot = np.arctan2(y0-y1, x0-x1)
        
        # return
        return pos_in_bot, direction_in_bot

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

def init_turtle():
    turtle = Turtle()
    def cleanup():
        turtle.set_twist(v=0, w=0)
    rospy.on_shutdown(cleanup)
    return turtle 
    
def main():
    
    classes = ["one", "two", "three", "four", "five"]

    fake_yolo_detector = FakeYoloDetector(classes=classes)
    voice_listener = VoiceListener()
    is_target_settled = False
    
    turtle = init_turtle()
    plane_detector = PlaneDetector()
    coord = CoordManager()
    
    ith_img = 0
    while (not is_target_settled) and (not rospy.is_shutdown()):
        print("="*80)

        # Get target object from speech command
        TARGET_OBJECT = voice_listener.wait_next_command(wait_time=1.0)
        if (not TARGET_OBJECT) or (TARGET_OBJECT not in classes):
            TARGET_OBJECT = "three"
            print("No input command. Set target as '{}', and keep in testing mode.".format(TARGET_OBJECT))
        else:
            is_target_settled = True
            print("TARGET OBJECT = {}".format(TARGET_OBJECT))
        
        # Get image
        ith_img += 1
        color_img, depth_img, cloud = plane_detector.sub_rgbd_and_cloud()
        img_disp = color_img.copy()
        
        # Detect plane and object
        p_cent, v_norm, p_target_2d, p_target_3d = [None]*4 # init output
        
        if True: # Detect plane, its center and norm
            
            p_cent, v_norm, cloud_plane = plane_detector.detect_plane(
                cloud, min_points=1200)
            
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
            
                if (p_target_3d is not None) and (v_norm is not None): # draw target object
                    img_disp = plane_detector.draw_plane_norm(
                        img_disp, p_target_3d, v_norm, color=[0, 255, 0])

        # Change coordinate
        pos_in_robot, direction_in_robot = coord.get_pos_and_dir_in_robot(
            p_in_cam=p_target_3d, 
            direction_in_cam=v_norm, 
            l_offset=0
        )
        if pos_in_robot is not None:
            cv2.putText(img_disp, "pos: {:.1f}, {:.1f}cm".format(*(pos_in_robot*100)), 
                        (50, 420), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=2, color=(0, 0, 255), thickness=2)
            cv2.putText(img_disp, "dir: {:.1f} degrees".format(direction_in_robot/np.pi*180), 
                        (50, 450), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=2, color=(0, 0, 255), thickness=2)
        # display
        print("p_cent = {}".format(p_cent))    
        print("v_norm = {}".format(v_norm))    
        print("p_target_2d = {}".format(p_target_2d))    
        print("p_target_3d = {}".format(p_target_3d))    
        print("pos_in_robot =       {}".format(pos_in_robot))    
        print("direction_in_robot = {}".format(direction_in_robot))    
        print("="*80)
        
        if not cv2_show(img_disp, ith_img, wait_key_ms=50):
            break 

        # save
        save_images(color_img, depth_img, img_disp)
        

        # end of object detection
        continue
    
    # -- Move robot
    print("\n\nMove robot to '{}'!!!\n".format(TARGET_OBJECT))
    turtle.reset_pose()
    
    # Re-compute goal position, with pencil length in consideration
    xy, theta = coord.get_pos_and_dir_in_robot(
        p_in_cam=p_target_3d, 
        direction_in_cam=v_norm, 
        l_offset=0.1, # distance between pen point and robot frame
    )
    print("target_pos   =       {}".format(xy))    
    print("target_theta = {}".format(theta))    
    turtle.move_to_relative_pose(xy[0], xy[1], theta)
    return 

if __name__ == '__main__':
    node_name = 'detect_plane_from_point_cloud'
    rospy.init_node(node_name)
    rospy.sleep(1)
    main()
    cv2.destroyAllWindows()
    