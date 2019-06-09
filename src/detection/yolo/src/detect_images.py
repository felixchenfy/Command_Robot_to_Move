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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import utils.lib_yolo_funcs as yolo
from utils.lib_common_funcs import Timer
from utils.lib_yolo_plot import Yolo_Detection_Plotter_by_cv2, Yolo_Detection_Plotter_by_very_slow_plt


# ===========================================================================


def set_arguments():
    parser = argparse.ArgumentParser()
    
    # Yolo args
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    
    # Args for doing inference
    parser.add_argument("--data_source", type=str, choices=['folder', 'video', 'webcam'], 
                        default="webcam", help="read data from a folder, video file, of webcam")
    parser.add_argument("--image_data_path", type=str, default="", 
                        help="depend on '--data_source', set this as: a folder, or a video file,")
     
    # Parse
    args = parser.parse_args()

    if 1: # add my arguments
            args.weights_path = 'weights/yolov3_ckpt_20.pth'
            args.model_def =    'data/digits_generated/yolo.cfg'
            args.class_path =   'data/digits_generated/classes.names'
            args.conf_thres = 0.95
            args.nms_thres = 0.1 # Remove a box if overlapping is larger than this 
            args.batch_size = 1
            args.n_cpu = 8
            args.data_source = "folder"
            args.image_data_path = 'data/digits_eval2/'
        
    return args 

    return args 


if __name__ == "__main__":
    
    # Set parameters
    args = set_arguments()
    IF_SHOW = True # if false, draw the image and save to file, but not show out
    IF_SINGLE_INSTANCE = True # single instance for each class
    
    # Check folders
    os.makedirs("output", exist_ok=True)
    
    # Init vars
    model = yolo.create_model(args)
    dataloader = yolo.set_dataloader(args)
    classes = yolo.load_classes(args.class_path)  # Extracts class labels from file

    # ------------------------- Start loop through images to detect ---------------
    
    # Set up plotting
    if 0: # very slow: 0.35s to plot. But the drawing is more beautiful.
        plotter = Yolo_Detection_Plotter_by_very_slow_plt(IF_SHOW)
    else: # fast: 0.02s to plot
        plotter = Yolo_Detection_Plotter_by_cv2(
            IF_SHOW=IF_SHOW, 
            cv2_waitKey_time=1, # wait keypress for 1ms
            resize_scale=1.5) # plot a larger image
        
    # Start detection
    print("\nPerforming object detection:")
    prev_time = time.time()
    cnt_img = 0
    
    for batch_i, (imgs_path, imgs) in enumerate(dataloader):

        # Detect
        # imgs: [B, W, H, 3], tensor, rgb
        imgs_detections = yolo.detect_targets(
            args, model, imgs, IF_SINGLE_INSTANCE)
        # Output:
        #   detections: [bbox, conf, cls_conf, cls_pred]
        #               bbox = [x1, y1, x2, y2] in the image coordinate
        
        # Log progress
        if 1:
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\nBatch %d, Inference Time: %s" % (batch_i, inference_time))
        
        # Draw "detections" onto each image
        for img_i, (img, path, detections) in enumerate(zip(imgs, imgs_path, imgs_detections)):
            
            img = img.numpy() # tensor -> numpy
            cnt_img += 1
            print("(%d) Image: '%s'" % (cnt_img, path))
            # timer_disp = Timer()
            
            # Plot
            plotter.plot(img, detections, classes)
            
            # Save
            filename = path.split("/")[-1].split(".")[0] # /usr/img.jpg --> img
            filename = f"output/{filename}.png"
            plotter.savefig(filename)

            # timer_disp.report_time(action="Display an image")
            
    plt.close()