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
    from utils.lib_geo_trans_ros import *
    
IN_SIMULATION = True

# ==================================================================================================
# Math
def xytheta_to_T(x, y, theta): # T: 3x3 transformation matrix
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.array([
        [c, -s, x,],
        [s,  c, y,],
        [0,  0, 1,],
    ])
    return T 

def T_to_xytheta(T): # T: 3x3 transformation matrix
    assert T.shape == (3, 3)
    x = T[0, 2]
    y = T[1, 2]
    s, c = T[1, 0], T[0, 0]
    theta = np.arctan2(s, c)
    return x, y, theta 
    
# ==================================================================================================

def call_service(service_name, service_type, args=None):
    rospy.wait_for_service(service_name)
    try:
        func = rospy.ServiceProxy(service_name, service_type)
        func(*args) if args else func()  # call this service
    except rospy.ServiceException as e:
        print("Failed to call service:", service_name)
        sys.exit()

# ==================================================================================================
# Control algorithms

class PIDcontroller(object):
    T = 1

    @classmethod
    def set_control_period(clf, T):
        PIDcontroller.T = T

    def __init__(self, P=0, I=0, D=0, dim=1):
        self.P = np.zeros(dim)+P
        self.I = np.zeros(dim)+I
        self.D = np.zeros(dim)+D
        self.err_inte = np.zeros(dim)
        self.err_prev = np.zeros(dim)
        self.dim = dim
        self.T = PIDcontroller.T

    def compute(self, err):
        out = 0
        err = np.array(err)

        # P
        out += np.dot(err, self.P)

        # I
        self.err_inte += err
        out += self.T * np.dot(self.err_inte, self.I)

        # D
        out += np.dot(err-self.err_prev, self.D) / self.T
        self.err_prev = err
        return out


def control_wheeled_robot_to_pose(
    turtle, x_goal, y_goal, theta_goal=None):
    # Reference: page 129 in "Robotics, Vision, and Control"

    # Robot config
    MAX_V = 0.1
    MAX_W = 0.6
    MIN_V = 0.0 # should be 0
    MIN_W = 0.0 # should be 0

    # Set control parameters
    T = 0.05  # control period
    PIDcontroller.set_control_period(T)
    exp_ratio = 1.0 # exp ratio applied to PID's output. should be 1
    k_vals = [0.3, 1.0, -0.5]
    # k_vals = [0.5, 1.2, -0.6]
    
    k_rho = k_vals[0] # reduce distance to the goal. P > 0
    k_alpha = k_vals[1] # drive robot towards the goal. P > P_rho
    if theta_goal is None: 
        theta_goal = 0
        k_beta = 0 # not considering orientation
    else:
        k_beta = k_vals[2] # make robot same orientation as desired. P < 0
                # 100% is too large
                    
    # Init PID controllers
    pid_rho = PIDcontroller(P=k_rho, I=0)
    pid_alpha = PIDcontroller(P=k_alpha, I=0)
    pid_beta = PIDcontroller(P=k_beta, I=0)

    # Loop and control
    cnt_steps = 0
    while not rospy.is_shutdown():
        cnt_steps += 1
        
        x, y, theta = turtle.get_pose()

        rho = calc_dist(x, y, x_goal, y_goal)
        alpha = pi2pi(math.atan2(y_goal - y, x_goal - x) - theta)
        beta = - theta - alpha + theta_goal


        # check direction
        sign = 1
        if abs(alpha) > math.pi/2:  # the goal is behind the robot
            alpha = pi2pi(math.pi - alpha)
            beta = pi2pi(math.pi - beta)
            sign = -1

        # Pass error into PID controller and obtain control output
        val_rho = pid_rho.compute(err=rho)[0]
        val_alpha = pid_alpha.compute(err=alpha)[0]
        val_beta = pid_beta.compute(err=beta)[0]

        # Get v and w 
        v = sign * val_rho**exp_ratio
        w = sign * (val_alpha + val_beta)**exp_ratio

        # Threshold on velocity
        v = max(MIN_V, min(abs(v), MAX_V)) * (1 if v > 0 else -1)  # limit v
        w = max(MIN_W, min(abs(w), MAX_W)) * (1 if w > 0 else -1) # limit w
        
        # Output
        turtle.set_twist(v, w)
        if cnt_steps % 10 == 0:
            turtle.print_state(x, y, theta, v, w)
            print("\trho = {}, alpha = {}, beta = {}".format(rho, alpha, beta))

        rospy.sleep(T)

        # Check stop condition
        if abs(x-x_goal)<0.01 and abs(y-y_goal)<0.01 and abs(theta-theta_goal)<0.1:
            break

    turtle.set_twist(v=0, w=0)
    print("Reach the target. Control completes.\n")


# ==================================================================================================
# Turtlebot class for other ROS node
class Turtle(object):
    def __init__(self):

        # Names
        self.model_name = "turtlebot3_waffle_pi"
        self.reference_frame = "world"

        # Pub
        self.pub_twist = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)
        
        # sub
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
        
        # Others
        Turtle.IN_SIMULATION = IN_SIMULATION
        
    def reset_global_pose(self):
        # set up the odometry reset publisher
        reset_odom = rospy.Publisher('/reset', Empty, queue_size=10)
        # reset odometry (these messages take a few iterations to get through)
        rospy.loginfo("Resetting robot state...")
        rospy.sleep(1.0)
        reset_odom.publish(Empty())
        rospy.sleep(1.0)
        rospy.loginfo("Resetting robot state... Complete")
        
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
        rostopic pub -r 20 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: turtlebot3_waffle_pi, pose: { position: { x: 1, y: 0, z: 2 }, orientation: {x: 0, y: 0.491983115673, z: 0, w: 0.870604813099 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'
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
        return True

    def move_a_line(self):
        while not rospy.is_shutdown():
            self.set_twist(v=0.1, w=0)
            x, y, theta = self.get_pose()
            self.print_state(x, y, theta)
            rospy.sleep(0.1)
        return True

    def move_to_pose(self, x_goal, y_goal, theta_goal=None):
        
        print("\nMove robot to the global pose: {}, {}, {}\n".format(
            x_goal, y_goal, theta_goal,))
  
        control_wheeled_robot_to_pose(
            self, x_goal, y_goal, theta_goal)
        
        return True
    
    def move_to_relative_pose(self, x_goal_r, y_goal_r, theta_goal_r):

        # convert x_goal from robot coordinate to the world coordinate
        x0, y0, theta0 = self.get_pose()
        Twr = xytheta_to_T(x0, y0, theta0) # T_world_to_robot
        Trg = xytheta_to_T(x_goal_r, y_goal_r, theta_goal_r) # T_robot_to_goal
        Twg = np.dot(Twr, Trg)       
        x_goal_w, y_goal_w, theta_goal_w = T_to_xytheta(Twg)

        # print("\nTwr: {}\nTwr: {}\nTwr: {}".format(Twr, Trg, Twg))     
        
        # move
        self.move_to_pose(x_goal_w, y_goal_w, theta_goal_w)
        
        return True
    
        # xg = x0 + distance * np.cos(theta0)
        # yg = x0 + distance * np.cos(theta0)
        
    # def move_forward(self, distance):
    #     x0, y0, theta0 = self.get_pose()
    #     xg = x0 + distance * np.cos(theta0)
    #     yg = x0 + distance * np.cos(theta0)