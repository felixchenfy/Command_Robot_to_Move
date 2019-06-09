''' This script runs in a python3+ env with torch installed.
A slave node will be started and waiting for the request from "yolo_request.py".
The master/slave communicate through a file defined in "../../config/config.py".

'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../config/" # ROS config folder
    sys.path.append(ROOT)
    import config
    
import time 
import yaml
import numpy as np 
import cv2 

from yolo.src.detect_one_image import Detector


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


class CommSlave(YamlFile):
    def __init__(self, filename):
        super(CommSlave, self).__init__(filename)
        
        # create the folder if not created
        folder = os.path.dirname(filename)
        if folder and (not os.path.exists(folder)):
            os.makedirs(folder)
            
    def wait_and_get_request(self, dt_check=0.05):
        while 1:
            time.sleep(dt_check)
            data  = self.read_yaml()
            
            try:
                if data["is_requested"]:
                    break
            except:
                Warning("Yolo slave node reads an empty yaml file: ", data)
                continue 
        data["is_requested"] = 0
        return data 
    
    def make_response(self, data):
        data["is_responded"] = 1
        self.write_yaml(data)        
    

class TrueYoloDetector(object):
    def __init__(self):
        self.args = config.set_args()
        self.filename = self.args["yolo_comm_txt"]
        self.image_name = self.args["yolo_comm_img"]
        self.slave = CommSlave(filename=self.filename)
        self.detector = Detector()
        
    def start(self):
        while 1:
            
            # wait request
            data = self.slave.wait_and_get_request(dt_check=0.05)
            
            # detect object in image
            img = cv2.imread(self.image_name, cv2.IMREAD_COLOR)
            detections = self.detector.detect(img)
            
            # write to data
            detections = detections.tolist() 
            data["detections"] = detections
            
            # response
            self.slave.make_response(data)
            

if __name__ == "__main__":
    true_yolo_detector = TrueYoloDetector()
    true_yolo_detector.start()
    