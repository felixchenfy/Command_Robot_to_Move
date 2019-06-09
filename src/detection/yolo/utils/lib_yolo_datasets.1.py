
''' Datasets for image yolo '''
'''
Part of this script if copied from "src/PyTorch_YOLOv3/utils/datasets.py" and then modified
'''

import glob 
import cv2 
import numpy as np 
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 
import time

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def rgbimg_to_yoloimg(img, img_size):
    
    # img = np.moveaxis(img, -1, 0)  # no need for this. torchvision.transforms does this for us.
    img = transforms.ToTensor()(img) # numpy, HxWx3 --> tensor, 3xHxW
    # img = img[np.newaxis, ...] # no need for this. DataLoader itself will add the additional channel.
        
    # Pad to square resolution
    img, _ = pad_to_square(img, 0) # 3 x H x W
    
    # Resize
    img = resize(img, img_size) # 3 x img_size x img_size

    return img

class ImgfolderDataset(Dataset):
    def __init__(self, folder_path, img_size=416, suffixes=("jpg", "png")):
        files = []
        for suffix in suffixes:
            files.extend( glob.glob(f"{folder_path}/*.{suffix}"))
        self.files = sorted(files)
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        
        # Extract image as PyTorch tensor
        if 1:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        else: # Alternatively, we can use: 
            img = np.array(Image.open(img_path)) # H x W x 3, RGB 
            # It returns a opened file of the img_path, 
            # and then use np.array to load in the data.
            
        # img = rgbimg_to_yoloimg(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)


class UsbcamDataset(object):
    ''' 
    Init "torch.utils.data.DataLoader" with an instance of this class,
    and then use enumerate() to get images.
    A complete test case is in "def test_usbcam"
    '''
    
    def __init__(self, img_size=416, max_framerate=10):
        self.cam = cv2.VideoCapture(0)
        self.img_size = img_size
        self.frame_period = 1.0/max_framerate*0.999
        self.prev_image_time = time.time() - self.frame_period
        self.cnt_img = 0
        
    def __len__(self):
        return 999999
    
    def __getitem__(self, index):

        # read next image
        self.wait_for_framerate()
        ret_val, img = self.cam.read()
        self.prev_image_time = time.time()
        self.cnt_img += 1
        img_path = "tmp/{:06d}.jpg".format(self.cnt_img)
        
        # change format for yolo
        # img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        img = rgbimg_to_yoloimg(img, self.img_size)
        return img_path, img
    
    def wait_for_framerate(self):
        t_curr = time.time()
        t_wait = self.frame_period - (t_curr - self.prev_image_time)
        if t_wait > 0:
            time.sleep(t_wait)

class VideofileDataset(object):
    ''' 
    Init "torch.utils.data.DataLoader" with an instance of this class,
    and then use enumerate() to get images.
    '''  
    def __init__(self, filename, img_size=416):
        self.cap = cv2.VideoCapture(filename)
        self.img_size = img_size
        self.cnt_img = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, index):
        ret_val, img = self.cap.read()
        self.cnt_img += 1
        img_path = "tmp/{:06d}.jpg".format(self.cnt_img)
        
        # change format for yolo
        # img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        img = rgbimg_to_yoloimg(img, self.img_size)
        return img_path, img
    
def test_usbcam():
    
    def tensor_images_to_list_numpy_images(input_tensor_imgs):
        imgs = input_tensor_imgs.permute(0, 2, 3, 1).data.numpy() # RGB, float, (20, H, W, 3)
        imgs = [img for img in imgs] # list of numpy image
        return imgs


    def cv2_plot(img, wait):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("", img) # imshow, needs BRG
        cv2.waitKey(wait)
    
    
    dataloader = DataLoader(
        UsbcamDataset(),
        batch_size=1,
        shuffle=False)
    time0 = time.time()
    
    for batch_i, (imgs_path, input_tensor_imgs) in enumerate(dataloader):
        print(time.time() - time0)
        imgs = tensor_images_to_list_numpy_images(input_tensor_imgs)
        for img in imgs:
            cv2_plot(img, wait=1)
    
    cv2.destroyAllWindows()
            
if __name__=="__main__":
    from torch.utils.data import DataLoader
    test_usbcam()