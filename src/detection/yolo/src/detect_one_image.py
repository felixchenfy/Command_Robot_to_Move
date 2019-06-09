if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import os
import sys
import time
import datetime
import argparse
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import types

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import utils.lib_yolo_funcs as yolo
from utils.lib_common_funcs import Timer
from utils.lib_yolo_plot import Yolo_Detection_Plotter_by_cv2, Yolo_Detection_Plotter_by_very_slow_plt
from utils.lib_plot import show 


# ===========================================================================


def set_arguments():
    args = types.SimpleNamespace()

    args.weights_path = ROOT + 'weights/yolov3_ckpt_20.pth'
    args.model_def =    ROOT + 'data/digits_generated/yolo.cfg'
    args.class_path =   ROOT + 'data/digits_generated/classes.names'
    args.conf_thres = 0.95
    args.nms_thres = 0.1 # Remove a box if overlapping is larger than this 
    args.batch_size = 1
    args.n_cpu = 8
    args.img_size = 416
    
    return args 

class Detector(object):
    ''' Yolo detector for single image '''
    def __init__(self):
        args = set_arguments()
        self.model = yolo.create_model(args)
        self.classes = yolo.load_classes(args.class_path)  # Extracts class labels from file
        self.plotter = Yolo_Detection_Plotter_by_cv2(IF_SHOW=True, 
                                                cv2_waitKey_time=0, resize_scale=1.5)
        self.args = args 
        
    def detect(self, 
               img,
               IF_SINGLE_INSTANCE = True, # single instance for each class
               
        ):
        # Change format to the required one
        imgs = self._cv2_format_to_detector(img) 
    
        # Detect
        imgs_detections = yolo.detect_targets(
            self.args, self.model, imgs, IF_SINGLE_INSTANCE)
        
        # Return
        detections = imgs_detections[0] # there is only 1 image here
        return detections 
    
    def plot(self, img, detections):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.plotter.plot(img, detections, self.classes)
    
    def _cv2_format_to_detector(self, img): # Output: [1, W, H, 3], tensor, rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = torch.from_numpy(img).unsqueeze(0)
        return imgs 
    
if __name__ == "__main__":
    
    # Load image
    img_filename = '/home/feiyu/catkin_ws/src/simon_says/src/detection/yolo/data/digits_eval3/image_1.png'
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    
    # Detect
    detector = Detector()
    detections = detector.detect(img)
    
    # Plot
    detector.plot(img, detections)
    
   