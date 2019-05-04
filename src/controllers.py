
import rospy
import math
from geo_trans import *

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

    # Set parameters
    T = 0.1  # control period
    PIDcontroller.set_control_period(T)
 
    k_rho = 0.5 # reduce distance to the goal. P > 0
    k_alpha = 1.0 # drive robot towards the goal. P > P_rho
    if theta_goal is None: 
        theta_goal = 0
        k_beta = 0 # not considering orientation
    else:
        k_beta = -0.3 # make robot same orientation as desired. P < 0
    
    # Init PID controllers
    pid_rho = PIDcontroller(P=k_rho, I=0)
    pid_alpha = PIDcontroller(P=k_alpha, I=0)
    pid_beta = PIDcontroller(P=k_beta, I=0)

    # Loop and control
    while not rospy.is_shutdown():

        x, y, theta = turtle.get_pose()

        rho = calc_dist(x, y, x_goal, y_goal)
        alpha = pi2pi(math.atan2(y_goal - y, x_goal - x) - theta)
        beta = - theta - alpha

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
        v = sign * val_rho
        w = sign * (val_alpha + val_beta)

        # Threshold on velocity
        v = min(abs(v), 0.3) * (1 if v > 0 else -1)  # limit v
        # w = min(abs(w), 0.5) * (1 if w>0 else -1) # limit w
        
        # Output
        turtle.set_twist(v, w)

        print("\n")
        turtle.print_state(x, y, theta, v, w)

        rospy.sleep(T)

    turtle.set_twist(v=0, w=0)
