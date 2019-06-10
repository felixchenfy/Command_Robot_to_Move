#!/usr/bin/env python

import argparse
import numpy as np
import sys
from time import time
import math

import rospy
from lib_turtlebot import Turtle 


parser = argparse.ArgumentParser()
parser.add_argument('--x', type=float, help='target x position',
                    default=0.5,)
parser.add_argument('--y', type=float, help='target y position',
                    default=0.5,)
parser.add_argument('--theta', type=float, help='target theta position',
                    default=0,)
args = parser.parse_args()

def test_PID_controller(args, turtle):
    try:
        # Set robot goal position
        if 0:
            x_goal, y_goal, theta_goal = 1, 1, 0
        else:
            x_goal, y_goal, theta_goal = args.x, args.y, args.theta

        rospy.loginfo("Robot goal position: x = {}, y = {}, theta = {}".format(
            x_goal, y_goal, theta_goal))
        rospy.sleep(1.0)

        # Move robot there
        turtle.move_to_pose(x_goal, y_goal, theta_goal)

    except rospy.ROSInterruptException:
        rospy.loginfo("ctrl+c detected")

def test_simple_move(args, turtle):
    turtle.move_a_line()
 
def test_complex_move(args, turtle):
    turtle.move_to_relative_pose(0.5, 0.5, np.pi/2) # P1: go half a circle
    turtle.move_to_relative_pose(0.5, 0.5, np.pi/2) # P2: go half a circle
    turtle.move_to_pose(0.5, 0.5, np.pi/2) # P3: go back to P1
    
def test_forward_and_backward(args, turtle):
    turtle.move_to_relative_pose(0.5, 0.0, 0) # forward
    turtle.move_to_relative_pose(-0.5, 0.0, 0) # backward
    turtle.move_to_relative_pose(0.0, 0.5, 0) # left

def main(args, turtle):
    ''' select one of the test function to test: '''
    
    # test_simple_move(args, turtle)
    # test_PID_controller(args, turtle)
    test_complex_move(args, turtle)
    # test_forward_and_backward(args, turtle)
    
    return 

if __name__ == '__main__':

    # init node
    rospy.init_node('main_move_turtle')
    rospy.loginfo("main_move_turtle node inits")

    # set node clean up
    def cleanup():
        turtle.set_twist(v=0, w=0)
    rospy.on_shutdown(cleanup)

    # init variables
    turtle = Turtle()
    
    # reset turtle's pose    
    rospy.sleep(0.2)
    if turtle.IN_SIMULATION:
        turtle.set_pose_in_simulation()
    else:
        turtle.reset_global_pose()
    rospy.sleep(0.2)
        
    # main
    main(args, turtle)
    rospy.spin()
    rospy.loginfo("Node ends")
