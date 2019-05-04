

import numpy as np 
import math
from tf.transformations import rotation_matrix
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf.transformations import euler_matrix, euler_from_matrix, quaternion_from_matrix, quaternion_matrix

def calc_dist(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)

def pi2pi(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

def quaternion_to_euler(quat_xyzw):
    
    def convert_quaternion_data_to_list(quat_xyzw):
        if type(quat_xyzw) != list and type(quat_xyzw) != np.ndarray:
            quat_xyzw=[quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w]
        return quat_xyzw

    quat_xyzw = convert_quaternion_data_to_list(quat_xyzw)
    euler_xyz = euler_from_quaternion(quat_xyzw, 'rxyz')
    return euler_xyz

def pose_to_xytheta(pose):
    x = pose.position.x
    y = pose.position.y
    euler_xyz = quaternion_to_euler(pose.orientation)
    theta = euler_xyz[2]
    return x, y, theta

def get_coordinate(ref_coord='/base_link', to_lookup_coord='/link_blank'):
    tf_listener = tf.TransformListener()
    try:
        (trans, rot) = tf_listener.lookupTransform(
            ref_coord, to_lookup_coord, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr("Error looking up transform")
    return None