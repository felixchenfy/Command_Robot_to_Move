
# Simon Says: Say a target, and the robot moves there.
Spring project at Northwestern University  
2019 April ~ June  
(This repo will be completed by 6/19)


# 1. Intro to the project

Hardware:  
* Turtlebot3-Waffle-Pi (a 2-wheel robot)  
* RGBD camera (Intel Realsense)
* A white board with numbers on it, including ①, ②, ③, ④, ⑤.
* My laptop as the main processor.

Procedures and goals: 
1. I put the Turtlebot robot on the ground.
2. I put the white board in front of the robot with some arbitrary pose.
3. I say a target number in English, such as "one" or "two". 
4. The robot detects the target number, and locates its position.
5. The robot moves to the white board and hits the target.

Key techniques:  
* Speech recognition (classification).
* Object detection to find targets.
* Point cloud processing for locating the target's pose.
* Control algorithm for moving robot to desired pose.

Environments:
* ROS Melodic on turtlebot with Python 2.
* ROS Melodic on my laptop with Python 2.
* Python 3 on my laptop for speech recognition and object detection. Communicate with ROS through file.

# 2. Done

## 2.0 Hardware setup
including: Turtlebot, realsense camera, communication with my laptop, etc.

## 2.1 Natural language proprocessing
I've trained an LSTM model to classify 10 types of audio segments, namely: "back", "five", "four", "front", "left", "one", "right", "stop", "three", "two".


## 2.2 Simple navigation algorithm

Given robot's current pose (x, y, theta) and a goal pose (x*, y*, theta*), I've implemented an algorithm that could drive the robot to the goal position based on a feedback control method.

## 2.3 Object detection by Yolo
Detect numbers 1~5 in RGB image.

## 2.4 Object pose estimation
First, detect plane in point cloud data.  
Then, from Yolo's result, we can locate the target's position and orientation.

# 3. TODO
1. Integrate the whole thing and take a demo video.  
    (Complete by June 14)
2. Update this README.   
    (Complete by June 18)

# 4. Main commands

see [doc/main_commands.md](doc/main_commands.md)

# 5. Other commands for testing

* Test robot controller in simulation

    In [lib_turtlebot.py](src/turtlebot_control/lib_turtlebot.py), set **IF_IN_SIMULATION** to True to test the robot in simulation.  
    In [move_turtle_to_pose.py](src/turtlebot_control/move_turtle_to_pose.py), in the **def main(** function, select a test case to test.

    Then:  
    > $ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch  
    > $ rosrun simon_says move_turtle_to_pose.py   

