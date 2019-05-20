

# Turtlebot3 Install


## Basics


### Gazebo
$ sudo apt-get install -y libgazebo9-dev
$ sudo apt-get install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
$ cd ~/catkin_ws/src
$ git clone https://github.com/ros-simulation/gazebo_ros_pkgs

Check missing dependencies:  
$ rosdep update  
$ rosdep check --from-paths . --ignore-src --rosdistro melodic  
Install anything missing:
```
apt	libnetpbm10-dev
apt	ros-melodic-hls-lfcd-lds-driver
apt	libgazebo9-dev
apt	ros-melodic-turtlebot3-applications-msgs
```

## Main

```
$ cd ~/catkin_ws/src
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3
$ cd turtlebot3
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs
$ git clone https://github.com/ROBOTIS-GIT/OpenCR
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations
# $ git clone https://github.com/ROBOTIS-GIT/turtlebot3_applications
# $ git clone https://github.com/ros-perception/ar_track_alvar
```

## Dependencies

$ sudo apt-get install ros-melodic-joy ros-melodic-teleop-twist-joy ros-melodic-teleop-twist-keyboard ros-melodic-laser-proc ros-melodic-rgbd-launch ros-melodic-depthimage-to-laserscan ros-melodic-rosserial-arduino ros-melodic-rosserial-python ros-melodic-rosserial-server ros-melodic-rosserial-client ros-melodic-rosserial-msgs ros-melodic-amcl ros-melodic-map-server ros-melodic-move-base ros-melodic-urdf ros-melodic-xacro ros-melodic-compressed-image-transport ros-melodic-rqt-image-view ros-melodic-navigation ros-melodic-interactive-markers

Gmapping has been released on ROS Melodic. The command below doesn't work:
$ sudo apt-get install ros-melodic-gmapping  
I need to install from source:
```
I followed this tutorial:
https://blog.csdn.net/wsc820508/article/details/81561304
create a folder under ~/catkin_ws/src/my_download_gmapping/, then:

git clone https://github.com/ros-perception/openslam_gmapping
git clone https://github.com/ros-perception/slam_gmapping.git
git clone https://github.com/ros-planning/navigation.git
git clone https://github.com/ros/geometry2.git
git clone https://github.com/ros-planning/navigation_msgs.git

```

### PCL
in python2.7 environment
conda install -c ccordoba12 python-pcl 


