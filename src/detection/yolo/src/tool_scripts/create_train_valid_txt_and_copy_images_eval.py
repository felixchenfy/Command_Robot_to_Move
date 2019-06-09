if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../../" # root of the project
    sys.path.append(ROOT)

import glob 
import os 
import numpy as np 
import sys, os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Settings
BASE_FOLDER = ROOT + "data/digits_generated/"
RATIO_valid_IMAGES = 0.2
EVAL_FOLDER_TRAIN = ROOT + "data/digits_eval_train/"
EVAL_FOLDER_VALID = ROOT + "data/digits_eval_valid/"

# Fixed settings
FILE_TRAIN = BASE_FOLDER + 'train.txt'
FILE_VALID = BASE_FOLDER + 'valid.txt'
FOLDER_IMAGES = BASE_FOLDER + 'images/'
FOLDER_LABELS = BASE_FOLDER + 'labels/'

def get_names(folder, file_types=('*.jpg', '*.png')):
    filenames = []
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

# Get image names
image_names = get_names(FOLDER_IMAGES, ('*.jpg', '*.png'))

# Check the corresponding label file of each image
if 0: # no need to do this

    def remove_path_in_name(name_with_path):
        if '/' in name_with_path:
            name = name_with_path.split('/')[-1]
        else:
            name = name_with_path
        return name

    def change_suffix(name, new_suffix=None):
        name = name[:name.rindex('.')]
        if new_suffix is None:
            return name 
        else:
            return name + "." + new_suffix
            
    label_names = []
    possible_label_names = get_names(FOLDER_LABELS, ('*.txt', ))
    for cnt_image, image_name in enumerate(image_names):
        # get corresponding label name of an image
        name = remove_path_in_name(image_name)
        name = change_suffix(name, new_suffix='txt')
        label_name = os.path.join(FOLDER_LABELS, name)

        if os.path.exists(label_name):
            data = np.loadtxt(label_name).reshape(-1, 5)
            print(data)
            if data.size != 0:
                pass # good
            else: # empty file, delete it
                os.remove(label_name)
            print("Image {} with {} objects. Filename = {}".format(
                    cnt_image, data.size//5, label_name))
        else:
            assert 0, f"{image_name} doesn't have label file"

# Write FILE_TRAIN and FILE_VALID
def train_valid_split(filenames, ratio_valid):
    N = len(filenames)
    n_train = int(N * (1 - ratio_valid))
    idxs = np.random.permutation(N)
    fname_trains = [filenames[i] for i in idxs[:n_train]]
    fname_valids = [filenames[i] for i in idxs[n_train:]]
    return fname_trains, fname_valids

def write_list_strings(filename, list_strings):
    with open(file=filename, mode='w') as f:
        for s in list_strings:
            f.write(s + "\n") 
    print("Wring {} strings to {}".format(len(list_strings), filename))

fname_trains, fname_valids = train_valid_split(image_names, RATIO_valid_IMAGES)
write_list_strings(FILE_TRAIN, fname_trains)
write_list_strings(FILE_VALID, fname_valids)

# Copy train/valid images to a new folder for later evaluation
from shutil import copyfile
def copy_files(src_filenames, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for name_with_path in src_filenames:
        name_only = name_with_path.split('/')[-1]
        copyfile(src=name_with_path, dst=dst_folder + name_only)
    print("Copy images to {}".format(dst_folder))
copy_files(fname_trains, EVAL_FOLDER_TRAIN)
copy_files(fname_valids, EVAL_FOLDER_VALID)