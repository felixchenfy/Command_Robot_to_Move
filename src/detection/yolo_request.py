#!/usr/bin/env python
''' This script is for ROS, and should run in Python2.7 '''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../config/" # ROS config folder
    sys.path.append(ROOT)
    import config
    
import time 
import yaml
import numpy as np 
import cv2 

from yolo.utils.lib_yolo_plot import Yolo_Detection_Plotter_by_cv2

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
    def __init__(self):
        self.args = config.set_args()
        self.filename = self.args["yolo_comm_txt"]
        self.image_name = self.args["yolo_comm_img"]
        self.master = CommMaster(filename=self.filename)

    def detect(self, img):
        master = self.master
        filename = self.filename

        # Write image to file for slave_node to detect         
        cv2.imwrite(self.image_name, img)
        
        # Make a request of detecting objects in image    
        master.make_request()
        print("-"*80)
        print("Master made a request")
            
        # wait and get result
        data = master.wait_and_get_response(dt_check=0.05)
        detections = np.array( data["detections"] )
        print("Master receives the response:")
        print("\t", detections)
        return detections
     
def detetions_to_label_and_pos(detections):
    res = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        label = classes[int(cls_pred)]
        pos = (int((x1+x2)/2), int((y1+y2)/2))
        res.append((label, pos))
    if 0: # test result
        for label, pos in res:
            print("Detect '{}', pos = {}".format(label, pos))
    return res 

if __name__=="__main__":
     # Load image
    img_filename = '/home/feiyu/catkin_ws/src/simon_says/src/detection/yolo/data/digits_eval3/image_1.png'
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

    # Set plotter
    plotter = Yolo_Detection_Plotter_by_cv2(
        IF_SHOW=True, cv2_waitKey_time=100, resize_scale=1.5)
    classes = ["one", "two", "three", "four", "five"]
    
    # Set up detector    
    fake_yolo_detector = FakeYoloDetector()
    for i in range(100):
        time.sleep(2)
        detections = fake_yolo_detector.detect(img)
        
        # Print detection result
        res = detetions_to_label_and_pos(detections)
        for label, pos in res:
            print("Detect '{}', pos = {}".format(label, pos))
            
        # Draw bbox of detected objects
        img_disp = img.copy()
        cv2.putText(img_disp, str(i), (50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, color=(0, 0, 0), thickness=2)
        img_disp = plotter.plot(img_disp, detections, classes, img_channels="bgr", if_print=False)
        
        
        
                
   