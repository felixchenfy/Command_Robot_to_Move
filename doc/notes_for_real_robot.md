
# Intro
See this page
http://emanual.robotis.com/docs/en/platform/turtlebot3/bringup/#load-a-turtlebot3-on-rviz

# On raspberry Pi

roslaunch turtlebot3_bringup turtlebot3_robot.launch

TIP: If you want to launch Lidar sensor, Raspberry Pi Camera, Intel® RealSense™ R200 or core separately, please use below commands.

$ roslaunch turtlebot3_bringup turtlebot3_lidar.launch
$ roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch
$ roslaunch turtlebot3_bringup turtlebot3_realsense.launch
$ roslaunch turtlebot3_bringup turtlebot3_core.launch

# Load a TurtleBot3 on Rviz

$ export TURTLEBOT3_MODEL=waffle_pi
$ roslaunch turtlebot3_bringup turtlebot3_remote.launch

One a new terminal window and enter the below command.
$ rosrun rviz rviz -d `rospack find turtlebot3_description`/rviz/model.rviz

(success)

# Topic Monitor
$ rqt
select the plugin -> Topics -> Topic Monitor

# Keyboard tele operation
http://emanual.robotis.com/docs/en/platform/turtlebot3/teleoperation/#keyboard
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch


# My program
rqt_graph

rosrun simon_says turtle_mover.py --x 0.4 --y 0.4 --theta 0
rosrun simon_says turtle_mover.py --x -0.4 --y -0.4 --theta -1.57
