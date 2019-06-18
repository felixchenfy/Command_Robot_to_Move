
# 2. Tests to run on my laptop  
> $ roscd simon_says

## 2.1 Test if connection is OK:  
> $ rostopic list  
> $ rostopic echo /odom  
> $ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

## 2.2 Test realsense camera
> $ roslaunch simon_says rviz_realsense.launch

## 2.3 Test control algorithm
In [src/turtlebot_control/lib_turtlebot.py](src/turtlebot_control/lib_turtlebot.py), set "IF_IN_SIMULATION = False".  
Then:
> $ rosrun simon_says move_turtle_to_pose.py

## 2.4 Use recorded images to replace RealSense Camera

rosrun this script: [src/realsense_io/fake_rgbd_image_publisher.py](src/realsense_io/fake_rgbd_image_publisher.py)  
It reads in color/depth images from the disk and publishes them with the same format as RealSense Camera.


# 3. Other commands for testing

* Test robot controller in simulation

    In [lib_turtlebot.py](src/turtlebot_control/lib_turtlebot.py), set "**IF_IN_SIMULATION** = True" to test the robot in simulation.  
    In [move_turtle_to_pose.py](src/turtlebot_control/move_turtle_to_pose.py), in the **"def main("** function, select a test case to test.

    Then:  
    > $ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch  
    > $ rosrun simon_says move_turtle_to_pose.py   