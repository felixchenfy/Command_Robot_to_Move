if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
from src.PyTorch_YOLOv3.models import Darknet
from src.PyTorch_YOLOv3.utils.utils import non_max_suppression, load_classes
from src.PyTorch_YOLOv3.utils.datasets import ImgfolderDataset

from utils.lib_yolo_datasets import ImgfolderDataset, UsbcamDataset, VideofileDataset
from utils.lib_common_funcs import Timer

import os
import sys
import time
import datetime
import argparse
import cv2 
import numpy as np 
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# ------------------------- Library functions -------------------------



def tensor_images_to_list_numpy_images(input_tensor_imgs):
    imgs = input_tensor_imgs.permute(0, 2, 3, 1).data.numpy() # RGB, float, (20, H, W, 3)
    imgs = [img for img in imgs] # list of numpy image
    return imgs
        
        
def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    ''' This is copied from src/PyTorch_YOLOv3/utils/utils.py '''
    
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

# ------------------ Main functions used for inference ------------------


def create_model(args):
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.model_def, img_size=args.img_size).to(device)

    # Load darknet weights
    if args.weights_path.endswith(".weights"):
        model.load_darknet_weights(args.weights_path)
    else: # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    model.eval()  # Set in evaluation mode

    return model 

def set_dataloader(args):

    print(f"Load data from: {args.data_source}; Data path: {args.image_data_path}")
    if args.data_source == "folder":
        dataloader = DataLoader(
            ImgfolderDataset(args.image_data_path, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=args.n_cpu, # This causes bug in vscode (threading problem). I comment this out during debug.
        )

    elif args.data_source == "video":

        dataloader = DataLoader(
            VideofileDataset(args.image_data_path, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    elif args.data_source == "webcam":
        dataloader = DataLoader(
            UsbcamDataset(max_framerate=10, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    else:
        raise ValueError("Wrong data source for yolo")  
    return dataloader


def detect_targets(args, model, imgs_path, input_tensor_imgs, if_single_instance=False):
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs_on_gpu = Variable(input_tensor_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        N_info = 7 # format of imgs_detections[jth_img]: x1, y1, x2, y2, conf, cls_conf, cls_pred
        imgs_detections = model(imgs_on_gpu)
        imgs_detections = non_max_suppression(imgs_detections, args.conf_thres, args.nms_thres)
        
        # convert to numpy array
        imgs_detections = [d.numpy() if d is not None else None for d in imgs_detections]

    # Get original images
    if 1: # this gives images of size [args.img_size, args.img_size]
        imgs = tensor_images_to_list_numpy_images(input_tensor_imgs)
    else: # This is wrong. 
        # The image for detection are padded, which is different from the original one.
        def tmp_func(filename):
            I = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            return I 
        imgs = [tmp_func(filename) for filename in imgs_path]
        
    # Sort detections based on confidence; 
    # Convert box to the current image coordinate
    for jth_img in range(len(imgs_detections)):
        if imgs_detections[jth_img] is None: continue
        detections = sorted(imgs_detections[jth_img], key=lambda x: x[5])
        detections = np.array(detections)
        detections = rescale_boxes(detections, args.img_size, imgs[jth_img].shape[:2])
        imgs_detections[jth_img] = detections
        
    # Remove duplicated objects
    # (under the assumption that each class has only one instance for each image)
    if if_single_instance:
        for jth_img, jth_detections in enumerate(imgs_detections):
            if imgs_detections[jth_img] is None: continue
            detected_objects = set()
            jth_unique_detections = []
            for kth_object in jth_detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = kth_object
                if cls_pred not in detected_objects: # Add object if not detected before
                    detected_objects.add(cls_pred)
                    jth_unique_detections.append(kth_object)
            imgs_detections[jth_img] = np.array(jth_unique_detections)
    
    return imgs, imgs_detections