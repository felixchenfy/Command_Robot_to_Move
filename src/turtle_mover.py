#!/usr/bin/env python

import rospy
import numpy as np
import sys

import roslib
import rospy
import math

import tf
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose, Twist, Vector3
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates

from geo_trans import *
from controllers import *

# Template for calling service


def call_service(service_name, service_type, args=None):
    rospy.wait_for_service(service_name)
    try:
        func = rospy.ServiceProxy(service_name, service_type)
        func(*args) if args else func()  # call this service
    except rospy.ServiceException as e:
        print("Failed to call service:", service_name)
        sys.exit()


class Turtle(object):
    def __init__(self):

        # Names
        self.model_name = "turtlebot3_waffle"
        self.reference_frame = "world"

        # Pub and sub
        self.pub_twist = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)
        self.sub_pose = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_sub_pose)

        # Robot state
        self.time0 = self.reset_time()
        self.pose = Pose()
        self.twist = Twist()
        
    def set_twist(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = -w
        self.pub_twist.publish(twist)

    def get_pose(self):
        x, y, theta = pose_to_xytheta(self.pose)
        return x, y, theta

    def set_pose(self, x=0, y=0, z=0):

        # Set robot state
        p = Point(x=x, y=y, z=z)
        q = Quaternion(x=0, y=0, z=0, w=0)
        pose = Pose(position=p, orientation=q)
        twist = Twist()
        state = ModelState(
            pose=pose, twist=twist,
            model_name=self.model_name, reference_frame=self.reference_frame)

        # Call service to set position
        call_service(
            service_name="/gazebo/set_model_state",
            service_type=SetModelState,
            args=(state, )
        )

        ''' Anathor way is to put the following code to command line:
        rostopic pub -r 20 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: turtlebot3_waffle, pose: { position: { x: 1, y: 0, z: 2 }, orientation: {x: 0, y: 0.491983115673, z: 0, w: 0.870604813099 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'
        '''

    def reset_time(self):
        self.time0 = rospy.get_time()
        return self.time0

    def query_time(self):
        return rospy.get_time()-self.time0

    def callback_sub_pose(self, model_states):
        idx = model_states.name.index(self.model_name)
        self.pose = model_states.pose[idx]
        self.twist = model_states.twist[idx]

    def print_state(self, x=None, y=None, theta=None, v=np.nan, w=np.nan):
        print("Robot pose: x = {:.3f}, y = {:.3f}, theta = {:.3f}, v = {:.3f}, w = {:.3f}".format(
            x, y, theta, v, w))

    def move_a_circle(self):
        while not rospy.is_shutdown():
            self.set_twist(v=0.1, w=0.1)
            print("Moving in circle ...")
            rospy.sleep(0.5)

    def move_to_pose(self, x_goal, y_goal ,theta_goal):
        
        if 1: # move to pose
            control_wheeled_robot_to_pose(
                self, x_goal, y_goal, theta_goal)
        
        elif 0: # move to point
            control_wheeled_robot_to_pose(
                self, x_goal, y_goal)



if __name__ == '__main__':

    # init node
    rospy.init_node('turtle_mover')

    # set node clean up
    def cleanup():
        turtle.set_twist(v=0, w=0)
    rospy.on_shutdown(cleanup)

    # init variables
    turtle = Turtle()
    turtle.set_pose()

    # run
    try:
        turtle.move_to_pose(x_goal=1, y_goal=1, theta_goal=0)
        # turtle.move_a_circle()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ctrl+c detected")
    rospy.loginfo("Node ends")
