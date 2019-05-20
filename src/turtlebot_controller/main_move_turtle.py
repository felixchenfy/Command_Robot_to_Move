#!/usr/bin/env python

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
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    from .lib_controllers import *
    from utils.lib_geo_trans_ros import *

# ==================================================================================================

def call_service(service_name, service_type, args=None):
    rospy.wait_for_service(service_name)
    try:
        func = rospy.ServiceProxy(service_name, service_type)
        func(*args) if args else func()  # call this service
    except rospy.ServiceException as e:
        print("Failed to call service:", service_name)
        sys.exit()


def reset_turtlebot():
    # set up the odometry reset publisher
    reset_odom = rospy.Publisher('/reset', Empty, queue_size=10)
    # reset odometry (these messages take a few iterations to get through)
    rospy.loginfo("Resetting robot state...")
    rospy.sleep(1.0)
    reset_odom.publish(Empty())
    rospy.sleep(1.0)
    rospy.loginfo("Resetting robot state... Complete")


IN_SIMULATION = False


class Turtle(object):
    def __init__(self):

        # Names
        self.model_name = "turtlebot3_waffle"
        self.reference_frame = "world"

        # Pub and sub
        self.pub_twist = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)
        if IN_SIMULATION:
            self.sub_pose = rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self.callback_sub_pose_simulation)
        else:
            self.sub_pose = rospy.Subscriber(
                "/odom", Odometry, self.callback_sub_pose)

        # Robot state
        self.time0 = self.reset_time()
        self.pose = Pose()
        self.twist = Twist()

    def set_twist(self, v, w):
        twist = Twist()
        twist.linear.x = v
        if IN_SIMULATION:
            twist.angular.z = -w
        else:
            twist.angular.z = w
        self.pub_twist.publish(twist)

    def get_pose(self):
        x, y, theta = pose_to_xytheta(self.pose)
        return x, y, theta
        # return x - self.x0, y - self.y0, theta - self.theta0

    # def reset_pose_offset(self):
    #     self.x0, self.y0, self.theta0 = pose_to_xytheta(self.pose)
    #     rospy.loginfo("Set pose offset: x0 = {}, y0 = {}, theta0 = {}".format(
    #         self.x0, self.y0, self.theta0))

    def set_pose_in_simulation(self, x=0, y=0, z=0):

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

        ''' Anathor way is to directly type following code in command line:
        rostopic pub -r 20 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: turtlebot3_waffle, pose: { position: { x: 1, y: 0, z: 2 }, orientation: {x: 0, y: 0.491983115673, z: 0, w: 0.870604813099 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'
        '''

    def reset_time(self):
        self.time0 = rospy.get_time()
        return self.time0

    def query_time(self):
        return rospy.get_time()-self.time0

    def callback_sub_pose_simulation(self, model_states):
        '''Callback function of "/gazebo/model_states" topic.'''
        idx = model_states.name.index(self.model_name)
        self.pose = model_states.pose[idx]
        self.twist = model_states.twist[idx]

    def callback_sub_pose(self, odometry):
        '''Callback function of "/odom" topic.'''
        # Contents:
        #   frame_id: "odom"
        #   child_frame_id: "base_footprint"
        self.pose = odometry.pose.pose
        self.twist = odometry.twist.twist
        # print(self.pose, self.twist)

    def print_state(self, x, y, theta, v=np.nan, w=np.nan):
        print("Robot pose: x = {:.3f}, y = {:.3f}, theta = {:.3f}, v = {:.3f}, w = {:.3f}".format(
            x, y, theta, v, w))

    def move_a_circle(self):
        while not rospy.is_shutdown():
            self.set_twist(v=0.1, w=0.1)
            print("Moving in circle ...")
            rospy.sleep(0.5)

    def move_a_line(self):
        while not rospy.is_shutdown():
            self.set_twist(v=0.1, w=0)
            x, y, theta = self.get_pose()
            self.print_state(x, y, theta)
            rospy.sleep(0.1)

    def move_to_pose(self, x_goal, y_goal, theta_goal):

        if 1:  # move to pose
            control_wheeled_robot_to_pose(
                self, x_goal, y_goal, theta_goal)

        elif 0:  # move to point
            control_wheeled_robot_to_pose(
                self, x_goal, y_goal)


parser = argparse.ArgumentParser()
parser.add_argument('--x', type=float, help='target x position')
parser.add_argument('--y', type=float, help='target y position')
parser.add_argument('--theta', type=float, help='target theta position')
args = parser.parse_args()


if __name__ == '__main__':

    # init node
    rospy.init_node('main_move_turtle')
    rospy.loginfo("main_move_turtle node inits")

    # set node clean up
    def cleanup():
        turtle.set_twist(v=0, w=0)
    rospy.on_shutdown(cleanup)

    reset_turtlebot()

    # init variables
    turtle = Turtle()
    rospy.sleep(0.2)
    if IN_SIMULATION:
        # Put robot to some pose. This is only for simulation mode
        turtle.set_pose_in_simulation()

    # run
    try:

        # Set robot goal position
        if 0:
            x_goal, y_goal, theta_goal = 1, 1, 0
        else:
            x_goal, y_goal, theta_goal = args.x, args.y, args.theta

        rospy.loginfo("Robot goal position: x = {}, y = {}, theta = {}".format(
            x_goal, y_goal, theta_goal))
        rospy.sleep(1.0)

        # Move robot to there
        turtle.move_to_pose(x_goal, y_goal, theta_goal)

    except rospy.ROSInterruptException:
        rospy.loginfo("ctrl+c detected")

    rospy.spin()
    rospy.loginfo("Node ends")
