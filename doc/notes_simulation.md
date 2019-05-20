


$ export TURTLEBOT3_MODEL=${waffle_pi} # burger, waffle, waffle_pi

# rviz
Start a rviz
$ roslaunch turtlebot3_fake turtlebot3_fake.launch

Tele operate
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch


# Gazebo

## Launch one of the world
Gazebo empty world:
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

Gazebo with a beautiful world:
roslaunch turtlebot3_gazebo turtlebot3_world.launch

Gazebo house
roslaunch turtlebot3_gazebo turtlebot3_house.launch

## Drive the robot

Me use keyboard to control the car:
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

Auto navigation by a package:
roslaunch turtlebot3_gazebo turtlebot3_simulation.launch

## Use Rviz to view sensor data
RViz visualizes published topics while simulation is running.

$ roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch


## SLAM
http://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#virtual-slam-with-turtlebot3

Launch Gazebo
$ export TURTLEBOT3_MODEL=waffle_pi
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch

Launch SLAM
$ export TURTLEBOT3_MODEL=waffle_pi
$ roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping

Remotely Control TurtleBot3
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

Save the Map
$ rosrun map_server map_saver -f ~/map


