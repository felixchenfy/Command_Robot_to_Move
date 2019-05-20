
import rospy
import math

if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../"
    sys.path.append(ROOT)
    from utils.lib_geo_trans_ros import *
    
# ==================================================================================================


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

    # Set control parameters
    T = 0.05  # control period
    PIDcontroller.set_control_period(T)
 
    k_rho = 0.3 # reduce distance to the goal. P > 0
    k_alpha = 1.0 # drive robot towards the goal. P > P_rho
    if theta_goal is None: 
        theta_goal = 0
        k_beta = 0 # not considering orientation
    else:
        k_beta = -0.5 # make robot same orientation as desired. P < 0
                # 100% is too large
                    
    # Init PID controllers
    pid_rho = PIDcontroller(P=k_rho, I=0)
    pid_alpha = PIDcontroller(P=k_alpha, I=0)
    pid_beta = PIDcontroller(P=k_beta, I=0)

    # Loop and control
    while not rospy.is_shutdown():

        x, y, theta = turtle.get_pose()

        rho = calc_dist(x, y, x_goal, y_goal)
        alpha = pi2pi(math.atan2(y_goal - y, x_goal - x) - theta)
        beta = - theta - alpha + theta_goal

        print("rho = {}, alpha = {}, beta = {}".format(rho, alpha, beta))

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
        v = min(abs(v), MAX_V) * (1 if v > 0 else -1)  # limit v
        w = min(abs(w), MAX_W) * (1 if w > 0 else -1) # limit w
        
        # Output
        turtle.set_twist(v, w)
        turtle.print_state(x, y, theta, v, w)

        rospy.sleep(T)

        # Check stop condition
        if abs(x-x_goal)<0.008 and abs(y-y_goal)<0.008 and abs(theta-theta_goal)<0.1:
            break

    turtle.set_twist(v=0, w=0)
    print("Reach the target. Control completes.\n")
