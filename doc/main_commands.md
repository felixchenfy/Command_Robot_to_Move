
# 1. Commands to start turtlebot 


## 1.1 On my laptop

Connect to turtlebot:  
```
Connect to the hidden wifi: RoboLab  
It's user name is: pi-cvlab
```

Set IP in ~/.bashrc:
```
export ROS_IP=192.168.1.204
```


Start ROS master:
$ roscore

Open another terminal and ssh into turtlebot
$ ssh pi@192.168.1.218  
$ sudo service vncserver@1 start

## 1.2 On turtlebot  

Add following to ~/.bashrc:  
```
export ROS_MASTER_URI=http://192.168.1.204:11311
export ROS_HOSTNAME=192.168.1.218
export ROS_IP=192.168.1.218
```

Launch sensors and bringups:  
pi@pi-cvlab:~$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

Launch realsense camera (if needed):
pi@pi-cvlab:~$ roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_infra1:=false enable_infra2:=false color_fps:=30 depth_fps:=30

Warning: Raspberry PI is slow. For realsense, please see this for solving error:  
https://github.com/IntelRealSense/realsense-ros/issues/669
https://github.com/IntelRealSense/realsense-ros  


# 2. Commands to run my main programs
$ roscd simon_says

## 2.0 Tests

### 2.0.1 Test if connection is OK:  
$ rostopic list  
$ rostopic echo /odom  
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

### 2.0.2 Test realsense camera
$ roslaunch simon_says rviz_realsense.launch

### 2.0.3 Test control algorithm
$ rosrun simon_says move_turtle_to_pose.py

## 2.1 Main

* Start script for **Object detection**:
$ roscd simon_says; open_anaconda; cd src/detection; python yolo_response.py

* Run main program detection
rosrun simon_says main.py 

* Start script for inputting **Speech Command**: 
$ roscd simon_says; open_anaconda; cd src/speech_recognition; python voice_speaker.py  

(After starting this, the keyboard key "R" should be pressed only when you want to record the audio)

Press key "R" to record audio and send your command to the laptop.


