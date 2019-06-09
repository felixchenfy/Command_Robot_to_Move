if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../" # root of the project
    sys.path.append(ROOT)

import cv2 
from config.config import read_all_args
args = read_all_args(config_file="config/config.yaml")

def read_list(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines 

# Get input images list
file_eval_images = args.f_yolo_valid
fnames = read_list(file_eval_images)

# Get output folder
folder_data_eval = args.f_data_eval
if not os.path.exists(folder_data_eval): os.makedirs(folder_data_eval)
print(f"Writing images to {folder_data_eval}")

# Copy images
for i, name in enumerate(fnames):
    print(f"{i}/{len(fnames)}: {name}")
    I = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(folder_data_eval + "/" + name.split('/')[-1], I)