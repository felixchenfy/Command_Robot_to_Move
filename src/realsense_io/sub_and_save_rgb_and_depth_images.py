#!/usr/bin/env python
# -*- coding: utf-8 -*-

if 1:  # common
    import numpy as np
    import copy
    import cv2
    from matplotlib import pyplot as plt
    from open3d import *
    import time
    import datetime
    import os
    import sys

if 1:  # ROS
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from std_msgs.msg import String

if 1:  # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    from utils.lib_geo_trans_ros import *
    from utils.lib_cloud import *
    from utils.lib_ros_topic import ColorImageSubscriber, DepthImageSubscriber
    from config.config import set_args
    
# ==================================================================================================

def get_time():
    s = str(datetime.datetime.now())[5:].replace(
        ' ', '-').replace(":", '-').replace('.', '-')[:-3]
    return s  # day, hour, seconds: 02-26-15-51-12-556


def write_images_to_file(color, depth, tmp_folder, suffix):
    if isinstance(suffix, int):
        suffix = "{:05d}".format(suffix)
    else:
        suffix = str(suffix)
    f1 = tmp_folder+"depth_"+suffix+".png"
    f2 = tmp_folder+"image_"+suffix+".png"
    cv2.imwrite(f1, depth)
    cv2.imwrite(f2, color)
    print("Save image {} to: {}".format(suffix, f1))

if __name__ == '__main__':
    node_name = 'read_image_from_realsense'
    rospy.init_node(node_name)

    # -------------------- Setings --------------------
    args = set_args()
    SAVE_IMAGES_TO = ROOT +'data/raw_rgbd_images/'

    # Keys
    KEY_STOP_PROGRAM = 'q'
    KEY_START_RECORD = 's'
    KEY_STOP_RECORD = 'd'
    KEY_TAKE_ONE_PHOTE = 'f'
    # --------------------------------------------------------
    
    # -- Init vars
    sub_color = ColorImageSubscriber(args["topic_color_image"])
    sub_depth = DepthImageSubscriber(args["topic_depth_image"])
    rospy.sleep(1)

    # Start recording =========================================================
    print(node_name+": node starts!!!")
    print(" === Press 's' to save image === ")

    is_recording = False
    while not rospy.is_shutdown():
        if(sub_color.isReceiveImage() and sub_depth.isReceiveImage):
            color, t1 = sub_color.get_image()
            depth, t2 = sub_depth.get_image()

            # draw
            depth_in_color = cv2.cvtColor(
                (depth/10).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            cv2.imshow("rgb + depth", np.hstack([color, depth_in_color]))
            
            key = max(cv2.waitKey(1), 0)
            key = chr(max(key, 0)) # convert from int to char

            if key == KEY_STOP_PROGRAM:
                break

            elif key == KEY_START_RECORD:
                if is_recording == False:
                    print("\n\n==========================")
                    print("Start recording\n")
                    is_recording = True
                    tmp_folder = SAVE_IMAGES_TO + "/" + get_time() + "/"
                    if not os.path.exists(tmp_folder):
                        os.mkdir(tmp_folder)
                    cnt_img_in_folder = 0

            elif key == KEY_STOP_RECORD:
                is_recording = False
                print("\n\n==========================")
                print("Stop recording\n")

            if is_recording == True:
                cnt_img_in_folder += 1
                write_images_to_file(color, depth, tmp_folder, cnt_img_in_folder)

            if key == KEY_TAKE_ONE_PHOTE:
                write_images_to_file(color, depth, SAVE_IMAGES_TO, get_time())

        rospy.sleep(0.01)

    plt.show()
    cv2.destroyAllWindows()
    print("Node stops")
