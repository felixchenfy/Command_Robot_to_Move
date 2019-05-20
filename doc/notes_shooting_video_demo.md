

* list all camera devices on ubuntu

$ ls -ltrh /dev/video*


* rosrun usb camera's node
$ rosrun usb_cam usb_cam_node _video_device:=/dev/video1 _pixel_format:=yuyv _camera_name:=tracker_camera
