

import numpy as np 
import cv2
import sys, os
import glob
import time

def create_folder(folder):
    print("Creating folder:", folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def get_filenames(folder, file_types=('*.wav',)):
    filenames = []
    
    if not isinstance(file_types, tuple):
        file_types = [file_types]
        
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

def get_dir_names(folder):
    names = [name for name in os.listdir(folder) if os.path.isdir(name)] 
    return names 

def get_all_names(folder):
    return os.listdir(folder)

def change_suffix(s, new_suffix, index=None):
    i = s.rindex('.')
    si = ""
    if index:
        si = "_" + str(index)
    s = s[:i] + si + "." + new_suffix
    return s 

def int2str(num, len):
    return ("{:0"+str(len)+"d}").format(num)

def add_idx_suffix(s, idx): # /data/two.wav -> /data/two_032.wav
    i = s.rindex('.')
    s = s[:i] + "_" + "{:03d}".format(idx) + s[i:]
    return s 

def cv2_image_float_to_int(img):
    img = (img*255).astype(np.uint8)
    row, col = img.shape
    rate = int(200 / img.shape[0])*1.0
    if rate >= 2:
        img = cv2.resize(img, (int(col*rate), int(row*rate)))
    return img

class Timer(object):
    def __init__(self):
        self.t0 = time.time()
    def report_time(self, event="", prefix=""):
        print(prefix + "Time cost of '{}' is: {:.2f} seconds.".format(
            event, time.time() - self.t0
        ))

if __name__=="__main__":
    print(change_suffix("abc.jpg", new_suffix='avi'))