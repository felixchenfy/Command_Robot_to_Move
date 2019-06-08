
from multiprocessing import Process, Value
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
    
class YamlFile(object):
    def __init__(self, filename):
        self.filename = filename 
    def read_yaml(self):
        return read_yaml(self.filename)
    def write_yaml(self, data):
        return write_yaml(self.filename, data)
    
class Detection_Master(YamlFile):
    def __init__(self, filename):
        super(Detection_Master, self).__init__(filename)
        self.init_yaml_file()
        
    def init_yaml_file(self):
        data = {
            "is_requested": 0,
            "is_responded": 0,
        }
        self.write_yaml(data)
        
    def make_request(self, request_key=None, request_value=None):
        data = self.read_yaml()
        data["is_requested"] = 1
        data["is_responded"] = 0
        if request_key:
            data[request_key] = request_value
        self.write_yaml(data)
        
    def wait_and_get_response(self, dt_check=0.05):
        while 1:
            time.sleep(dt_check)
            data = self.read_yaml()
            if data["is_responded"]:
                break 
        return data
         
class Detection_Servant(YamlFile):
    def __init__(self, filename):
        super(Detection_Servant, self).__init__(filename)
        
    def wait_and_get_request(self, dt_check=0.05):
        while 1:
            time.sleep(dt_check)
            data  = self.read_yaml()
            if data["is_requested"]:
                break 
        data["is_requested"] = 0
        return data 
    
    def make_response(self, data):
        data["is_responded"] = 1
        self.write_yaml(data)        
    
# ==================================================

def node_master():
    master = Detection_Master(filename='communications.yaml')
    while 1:
        time.sleep(2)
        
        # request
        master.make_request(request_key="filename", request_value="image.jpg")
        print("-"*80)
        print("Master made a request")
        
        # wait and get result
        data = master.wait_and_get_response(dt_check=0.05)
        print("Master receives the response:")
        print("\t", data["bboxes"])
        
def node_servant():
    servant = Detection_Servant(filename='communications.yaml')
    while 1:
        
        # wait request
        data = servant.wait_and_get_request(dt_check=0.05)
        
        # process
        data["bboxes"] = data["filename"] + ": " + str(np.random.random())
        
        # response
        servant.make_response(data)

if __name__=="__main__":
    
    thread1 = Process(target=node_master, args=())
    thread1.start()
    thread2 = Process(target=node_servant, args=())
    thread2.start()

    # wait to quit
    thread1.join()
    thread2.join()
    print("thread ends")
