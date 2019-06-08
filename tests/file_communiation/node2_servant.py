
import time 
import yaml
import numpy as np 

def read_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

def write_yaml(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
def process_data(data):
    if not data["is_image_ready"]:
        return 
    data["is_image_ready"] = 0
    data["is_detection_completed"] = 1
    data["bboxes"] = [np.random.random()]
    write_yaml(filename, data)
    
filename = 'communications.yaml'
for i in range(200):
    time.sleep(0.2)
    data = read_yaml(filename)
    process_data(data)