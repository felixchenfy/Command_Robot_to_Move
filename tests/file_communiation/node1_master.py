
import time 
import yaml

data = dict(
    A = 'a',
    B = dict(
        C = 'c',
        D = 'd',
        E = 'e',
    )
)

def read_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

def write_yaml(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
filename = 'communications.yaml'
data = read_yaml(filename)
for i in range(200):
    time.sleep(1) # for every 1 second, send a request
    print(i)
    
    # Send request
    data["is_image_ready"] = 1
    data["is_detection_completed"] = 0
    write_yaml(filename, data)
    print("Send a request")
    
    # Wait for response
    while 1:
        time.sleep(0.05)
        data = read_yaml(filename)
        if data["is_detection_completed"]:
            break
    
    # Display response
    print("The response is", data)