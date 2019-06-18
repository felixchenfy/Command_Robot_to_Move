
# 1. Commands to start Turtlebot 


## 1.1 On my laptop

Connect to Turtlebot:  
```
Connect to the hidden wifi: RoboLab  
User name on Turtlebot's Ubuntu: pi-cvlab
```

Set my laptop's IP in ~/.bashrc:   
```
export ROS_IP=192.168.1.204
```


Start ROS master:  
> $ roscore

Open another terminal and ssh into Turtlebot:  
> $ ssh pi@192.168.1.218   

You may or may not open vncserver:
> $ sudo service vncserver@1 start

## 1.2 On Turtlebot  

Add following to ~/.bashrc:   
```
export ROS_MASTER_URI=http://192.168.1.204:11311
export ROS_HOSTNAME=192.168.1.218
export ROS_IP=192.168.1.218
```

Launch sensors and bringups:  
> pi@pi-cvlab:~$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

Launch realsense camera (if needed):   
> pi@pi-cvlab:~$ roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_infra1:=false enable_infra2:=false color_fps:=30 depth_fps:=30

Notes: Raspberry PI is slow. For Realsense, please see this for solving error:    
https://github.com/IntelRealSense/realsense-ros/issues/669
https://github.com/IntelRealSense/realsense-ros  
