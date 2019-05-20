import numpy as np
import tf
import cv2
import math
import rospy
from tf.transformations import rotation_matrix
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf.transformations import euler_matrix, euler_from_matrix, quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import Pose, Point, Quaternion


if 0: # Functions below has already been defined in lib_geo_trans
    def rotx(angle, matrix_len=4):
        xaxis=(1, 0, 0)
        return rotation_matrix(angle, xaxis)

    def roty(angle, matrix_len=4):
        yaxis=(0, 1, 0)
        return rotation_matrix(angle, yaxis)
        
    def rotz(angle, matrix_len=4):
        zaxis=(0, 0, 1)
        return rotation_matrix(angle, zaxis)

# a bit wrap for geometry_msgs.msg.Pose
def toRosPose(pos, quaternion):
    if(type(pos)==list or type(pos) == np.ndarray):
        pos = Point(pos[0],pos[1],pos[2])
    if(type(quaternion)==list or type(quaternion) == np.ndarray):
        quaternion = Quaternion(quaternion[0],quaternion[1],quaternion[2],quaternion[3])
    return Pose(pos, quaternion)

# ROS pose to T4x4
def pose2T(pos, quaternion):
    # Trans to 4x4 matrix
    if(type(pos)!=list and type(pos) != np.ndarray):
        pos = [pos.x, pos.y, pos.z]
    R = quaternion_to_R(quaternion)
    T = form_T(R, pos)
    return T

def list_to_quat(l):
    quat=Quaternion(l[0],l[1],l[2],l[3])
    return quat

def quaternion_to_R(quat_xyzw):
    if type(quat_xyzw) != list and type(quat_xyzw) != np.ndarray:
        quat_xyzw=[quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w]
    if 0:
        euler_xyz = euler_from_quaternion(quat_xyzw, 'rxyz')
        R = euler_matrix(euler_xyz[0], euler_xyz[1],
                        euler_xyz[2], 'rxyz')[0:3, 0:3]
    else:
        R = quaternion_matrix(quat_xyzw)[:3,:3]
    return R

def Rp_to_pose(R, p):
    if R.shape[0]==3: # expand the matrix to 4x4
        tmp=np.identity(4)
        tmp[0:3,0:3]=R
        R=tmp
    quaternion = quaternion_from_matrix(R)
    return Pose(p ,quaternion)

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

if __name__=="__main__":

    # -- Prepare data
    p = [1,2,3]
    euler = [ 0.3, 0.5, 1.0]
    R = euler_matrix(euler[0],euler[1],euler[2], 'rxyz')[0:3,0:3]
    quaternion = quaternion_from_euler(
        euler[0],euler[1],euler[2]) # [0.24434723 0.1452622  0.4917509  0.82302756]
    
    # -----------------------------------------------------------------------
    # -- Copy test case below

    # ================================
    # print(form_T(R, p))

    # ================================
    # print(R)
    # print(quaternion_to_R(quaternion))

    # ================================
    # print(quaternion)
    # print(toRosPose(p, quaternion))

    # ================================
    # print(Rp_to_pose(R, p))
