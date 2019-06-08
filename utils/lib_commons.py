# -*- coding: utf-8 -*-

import numpy as np
import cv2 

def list2str(vals):
    str_val = ["{:.3f} ".format(val) for val in vals]
    str_val = ", ".join(str_val)
    return str_val
    
def savetxt(filename, data):
    np.savetxt(filename, data, delimiter=",")

def loadtxt(filename):
    return np.loadtxt(filename, delimiter=",")

def increase_color(img, mask, color):
    '''
    img: 3 channels, uint8
    mask: 1 channel, same size as img, bool
    color: a list of 3 values
    '''
    img = img.astype(np.int32)
    for i in range(3):
        imgi = img[:, :, i]
        imgi[mask] += color[i]
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    return img