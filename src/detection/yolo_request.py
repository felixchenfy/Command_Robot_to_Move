#!/usr/bin/env python
''' This script is for ROS, and should run in Python2.7 '''

from yolo.utils.lib_yolo_plot import Yolo_Detection_Plotter_by_cv2

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../" # ROS config folder
    sys.path.append(ROOT)
    import config.config as config 
    
import time 
import yaml
import numpy as np 
import cv2 
import os

class YamlFile(object):
    def __init__(self, filename):
        self.filename = filename 

    def read_yaml(self):
        with open(self.filename, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def write_yaml(self, data):
        with open(self.filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

class CommMaster(YamlFile):
    def __init__(self, filename):
        super(CommMaster, self).__init__(filename)
        
        # create the folder if not created
        folder = os.path.dirname(filename)
        if folder and (not os.path.exists(folder)):
            os.makedirs(folder)
        
        self.init_yaml_file()
        
    def init_yaml_file(self):
        data = {
            "is_requested": 0,
            "is_responded": 0,
        }
        self.write_yaml(data)
        
    def make_request(self):
        data = self.read_yaml()
        data["is_requested"] = 1
        data["is_responded"] = 0
        self.write_yaml(data)
        
    def wait_and_get_response(self, dt_check=0.05):
        while 1:
            time.sleep(dt_check)
            data = self.read_yaml()
            try:
                if data["is_responded"]:
                    break
            except:
                Warning("Yolo master node reads an empty yaml file: ", data)
                continue 
        return data

class FakeYoloDetector(object):
    def __init__(self, classes):
        self.args = config.set_args()
        self.filename = self.args["yolo_comm_txt"]
        self.image_name = self.args["yolo_comm_img"]
        self.master = CommMaster(filename=self.filename)
        self.plotter = Yolo_Detection_Plotter_by_cv2(IF_SHOW=False)
        self.classes = classes
        
    def draw_detections_onto_image(
            self, 
            img_disp, 
            detections, 
            img_channels="bgr", 
            if_print=False,
        ):
        img_disp = self.plotter.plot(
            img_disp, detections, self.classes, img_channels=img_channels, if_print=if_print)
        return img_disp
    
    def detect(self, img, if_print=True):
        master = self.master
        filename = self.filename

        # Write image to file for slave_node to detect         
        cv2.imwrite(self.image_name, img)
        
        # Make a request of detecting objects in image    
        master.make_request()
        if if_print:
            print("-"*80)
            print("Master made a request")
            
        # wait and get result
        data = master.wait_and_get_response(dt_check=0.05)
        detections = np.array(data["detections"])
        if if_print:
            print("Master receives the response:")
            print("\t", detections)
        return detections
     
    def cv2_show(self, img_disp, wait_key_ms):
        cv2.imshow("", img_disp)
        q = cv2.waitKey(wait_key_ms)
        return q 
    
    def detetions_to_labels_and_pos(self, detections):
        ''' 
        Input:
            detections: the output of "detect_targets()" 
        '''
        labels_and_pos = []
        classes = self.classes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            label = classes[int(cls_pred)]
            pos = (int((x1+x2)/2), int((y1+y2)/2))
            labels_and_pos.append((label, pos))
        if 0: # test result
            for label, pos in labels_and_pos:
                print("Detect '{}', pos = {}".format(label, pos))
        return labels_and_pos 

if __name__=="__main__":
     # Load image
    img_filename = '/home/feiyu/catkin_ws/src/simon_says/src/detection/yolo/data/digits_eval3/image_1.png'
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

    
    # Set up detector    
    classes = ["one", "two", "three", "four", "five"]
    fake_yolo_detector = FakeYoloDetector(classes)
    
    for i in range(100):
        time.sleep(2)
        detections = fake_yolo_detector.detect(img)
        
        # Print detection result
        labels_and_pos = fake_yolo_detector.detetions_to_labels_and_pos(detections)
        for label, pos in labels_and_pos:
            print("Detect '{}', pos = {}".format(label, pos))
            
        # Draw bbox of detected objects
        img_disp = img.copy()
        cv2.putText(img_disp, str(i), (50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, color=(0, 0, 0), thickness=2)
        img_disp = fake_yolo_detector.draw_detections_onto_image(
            img_disp, detections, classes)
        
        # Show
        fake_yolo_detector.cv2_show(img_disp, wait_key_ms=100)
        
        
                
   