
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
# from lib_geo_trans import transXYZ, rotx, roty, rotz, world2pixel

def show(imgs, figsize=(6, 10), layout=None):
    
    def convert(img):
        '''change image color from "BGR" to "RGB" for plt.plot()'''
        if isinstance(img.flat[0], np.float):
            img = (img*255).astype(np.uint8)
        if len(img.shape)==3 and img.shape[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    # Check input
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    imgs = [convert(img) for img in imgs]

    # Init figure
    plt.figure(figsize=figsize)
    
    # Set subplot size
    N = len(imgs)
    if layout is not None:
        r, c = layout[0], layout[1]
    else:
        if N <= 4:
            r, c = 1, N
        else:
            r, c = N//4+1, 4

    # Plot
    for i in range(N):
        plt.subplot(r, c, i+1)
        plt.imshow(imgs[i])
    plt.show()

def cv2_imshow(img):
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plot_3d_cloud(xyz):
    from mpl_toolkits.mplot3d import Axes3D  # THIS MUST BE IMPORTED!
    fig = plt.figure().gca(projection='3d')
    fig.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
    
def draw_bbox(img, bbox, color=(0,255,0), thickness=2):
    # img = img.copy()
    r, c = img.shape[:2]
    x, y, w, h = bbox  
    x0 = (x - w / 2) * c
    x1 = (x + w / 2) * c
    y0 = (y - h / 2) * r
    y1 = (y + h / 2) * r 
    x0, x1, y0, y1 = map(int, [x0, x1, y0, y1])
    img = cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        color=color,
        thickness=thickness)
    # return img
    
    

def plot_cloud_3d(
        xyz, # Nx3 or 3xN
        alpha=None, # N
        color=None, # Nx3 or 3xN
        figsize=(12, 12), title='', ax=None,
        
        # image size
        w=640,  h=480, 
        camera_intrinsics=None,
        
        # view angle
        X = 0, 
        Y = 0,
        Z = 0,
        ROTX = 0,
        ROTY = 0,
        ROTZ = 0,
        
        # Plot settings
        point_scale = 2,
    ):
    ''' Project 3d point cloud onto 2d image, and display'''

    # Check input
    if xyz.shape[0] != 3:
        xyz = xyz.T # 3xN
    if color is not None and color.shape[1] != 3:
        color = color.T # Nx3
    if color is not None:
        # color = np.column_stack( (color[:, 2], color[:, 1], color[:, 0]))  # bgr --> rgb
        pass 
    
    # Create figure axes
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    # Camera intrinsics
    if camera_intrinsics is None:
        camera_intrinsics = np.array([
            [w, 0, w/2],
            [0, h, h/2],
            [0, 0,   1]
        ], dtype=np.float32)
    
    # Set view angle, and transformation matrix
    # ...
    T_world_to_camera = transXYZ(x=X, y=Y, z=Z).dot(
        rotz(ROTZ)).dot(rotx(ROTX)).dot(roty(ROTY))
    T_cam_to_world = np.linalg.inv(T_world_to_camera)

    # Transform points' world positions to image pixel positions
    p_world = xyz
    p_image = world2pixel(p_world, T_cam_to_world, camera_intrinsics)
    # to int, so it cloud be plot onto image
    p_image = np.round(p_image).astype(np.int)

    # Put each point onto image
    N = xyz.shape[1] # number of points
    w = int(w/point_scale)
    h = int(h/point_scale)
    zeros, ones = np.zeros((h, w)), np.ones((h, w))
    image = np.zeros((h, w, 3))
    
    if color is not None:
        for i in range(N):  # iterate through all points
            u, v, colors = p_image[0, i], p_image[1, i], color[i]
            u = int(u/point_scale)
            v = int(v/point_scale)
            if w > u >= 0 and h > v >= 0:
                image[v][u] = colors

    elif alpha is not None:
        # TODO: DEBUG THIS
        pass
        # alpha = np.ones((N, )) 
        # for i in range(N):  # iterate through all points
        #     x, y, a = p_image[0, i], p_image[1, i], alpha[i]
        #     u, v = y, x  # flip direction to match the plt plot
        #     if w > u >= 0 and h > v >= 0:
        #         image[v][u][0] = max(image[v][u][0], a)
        #         image[v][u][2] = 1 - image[v][u][0]

    # Show
    ax.imshow(image)
    plt.axis('off')
    
    
    
    
    
    
    
    
    
    
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# Copied from lib_geo_trans:


'''
Geometric and camera related transformations
'''

import numpy as np
import copy
import cv2


def form_T(R, p):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = np.array(p).reshape((3, 1))
    return T


def get_Rp_from_T(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]
    return (R, p)


def invRp(R, p):
    T = form_T(R, p)
    T = np.linalg.inv(T)
    R_inv, p_inv = get_Rp_from_T(T)
    return R_inv, p_inv


def transXYZ(x=None, y=None, z=None):
    T = np.identity(4)
    data = [x, y, z]
    for i in range(3):
        if data[i] is not None:
            T[i, 3] = data[i]
    return T


def rot3x3_to_4x4(R):
    T = np.identity(4)
    T[0:3, 0:3] = R
    return T


def rot(axis, angle, matrix_len=4):
    R_vec = np.array(axis).astype(float)*angle
    R, _ = cv2.Rodrigues(R_vec)
    if matrix_len == 4:
        R = rot3x3_to_4x4(R)
    return R


def rotx(angle, matrix_len=4):
    return rot([1, 0, 0], angle, matrix_len)


def roty(angle, matrix_len=4):
    return rot([0, 1, 0], angle, matrix_len)


def rotz(angle, matrix_len=4):
    return rot([0, 0, 1], angle, matrix_len)


def euler2matrix(x, y, z, order='rxyz'):
    return rotx(x).dot(roty(y)).dot(rotz(z))


# ----------------Point's pos transformation between world/camera/image

# Distort a point. Input x,y are the point's pos on the camera normalized plane (z=0)


def distortPoint(x, y, distortion_coeffs):
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    d = distortion_coeffs
    k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]
    x_distort = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
        2*p1*x*y + p2*(r2 + 2*x*x)
    y_distort = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
        p1*(r2 + 2*y*y) + 2*p2*x*y
    pt_cam_distort = np.array([[x_distort, y_distort, 1]]).transpose()
    return pt_cam_distort

# Represent a point from using world coordinate to using camera coordinate


def world2cam(p, T_cam_to_world):
    if type(p) == list:
        p = np.array(p)
    if p.shape[0] == 3:
        p = np.vstack((p, np.ones((1, p.shape[1]))))
    p_cam = T_cam_to_world.dot(p)
    return p_cam[0:3, :]

# Project a point represented in camera coordinate onto the image plane


def cam2pixel(p, camera_intrinsics, distortion_coeffs=None):

    # Transform to camera normalized plane (z=1)
    p = p/p[2, :]  # z=1

    # Distort point
    if distortion_coeffs is not None:
        for i in range(p.shape[1]):
            p[0:2, i] = distortPoint(p[0, i], p[1, i], distortion_coeffs)
            assert 0, "TODO: I haven't tested this for loop"

    # Project to image plane
    pt_pixel_distort = camera_intrinsics.dot(p)
    return pt_pixel_distort[0:2, :]

# A combination of the above two

def world2pixel(p, T_cam_to_world, camera_intrinsics, distortion_coeffs=None):
    return cam2pixel(world2cam(p, T_cam_to_world), camera_intrinsics, distortion_coeffs)

