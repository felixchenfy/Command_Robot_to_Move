

if 1: # Set path
    import sys, os
    ROS_ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../config/" # ROS config folder
    sys.path.append(ROS_ROOT)
    from config import set_args
    
import time 
import yaml
import numpy as np 
import cv2 

class VoiceListener(object):
    def __init__(self):
        self.args = set_args()
        filename = self.args["voice_comm_txt"]
        self.filename = filename
        
        # create the folder if not created
        folder = os.path.dirname(filename)
        if folder and (not os.path.exists(folder)):
            os.makedirs(folder)
        
        # init communication file
        data = {
            "new_command_updated": 0,
            "command_label": "",
        }
        self._write_yaml(data)
        
    def wait_next_command(self, dt_check=0.05):
        
        # wait for new command
        while 1:
            time.sleep(dt_check)
            data  = self._read_yaml() # TODO
            try:
                if data["new_command_updated"]:
                    break
            except:
                Warning("Voice listener reads an empty yaml file: ", data)
                continue 
        
        # update listener state
        data["new_command_updated"] = 0
        self._write_yaml(data)
        
        # return command label
        command_label = data["command_label"]
        return command_label 
                 
    def _read_yaml(self):
        with open(self.filename, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def _write_yaml(self, data):
        with open(self.filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    voice_listerner = VoiceListener()
    while 1:
        print("\n")
        print("Start listening to command")    
        command_label = voice_listerner.wait_next_command()
        print("Receive a command: {}".format(command_label))
        