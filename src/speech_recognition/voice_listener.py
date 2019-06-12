

if 1: # Set path
    import sys, os
    ROS_ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../" # ROS config folder
    sys.path.append(ROS_ROOT)
    from config.config import set_args
    
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
        
    def wait_next_command(self, dt_check=0.05, wait_time=1.0):
        
        # wait for new command
        N = int(wait_time / dt_check)
        flag = False
        for i in range(N):
            time.sleep(dt_check)
            data  = self._read_yaml() # TODO
            try:
                if data["new_command_updated"]:
                    flag = True
                    break
            except:
                Warning("Voice listener reads an empty yaml file: ", data)
                continue 
            
        if flag: # no command is received
            # update listener state
            data["new_command_updated"] = 0
            self._write_yaml(data)
            
            # return command label
            command_label = data["command_label"]
            return command_label 
        else:
            return None 
                         
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
        print("\nStart listening to command")    
        command_label = voice_listerner.wait_next_command(wait_time=1.0)
        if command_label is None:
            print("Receive no command.")
        else:
            print("Receive a command: {}".format(command_label))
        time.sleep(1.0)