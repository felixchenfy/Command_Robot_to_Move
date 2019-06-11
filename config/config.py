
if 1: # my lib
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    
import yaml 

def set_args(filename="config.yaml"):
    
    # args = load_yaml(filename)
    args = {}
    
    args["topic_color_image"] = "/camera/color/image_raw"
    args["topic_depth_image"] = "/camera/aligned_depth_to_color/image_raw"
    args["topic_camera_info"] = "/camera/color/camera_info"
    args["filename_camera_info"] =  ROOT + "config/cam_params_realsense.json"
  
    args["yolo_comm_txt"] = ROOT + "comm/yolo_comm.txt"
    args["yolo_comm_img"] = ROOT + "comm/yolo_comm.png"

    args["voice_comm_txt"] = ROOT + "comm/voice_comm.txt"

    return args

def load_yaml(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded