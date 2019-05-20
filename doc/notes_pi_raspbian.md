


# =====================================
IP related things


# Tutorials

* Set up ethernet  
https://stackoverflow.com/questions/16040128/hook-up-raspberry-pi-via-ethernet-to-laptop-without-router

* Connect to wifi
https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md
https://stackoverflow.com/questions/37312501/how-do-i-detect-and-connect-to-a-hidden-ssid-on-my-raspiberry-pi-3-raspbian


# Remote PC (my laptop)

* Show device under same ethernet
cat /var/lib/misc/dnsmasq.leases

* Show device under same local network
    method 1: use ipscan software

* Set static IP 
https://raspberrypi.stackexchange.com/questions/37920/how-do-i-set-up-networking-wifi-static-ip-address/74428#74428


* SSH
ssh pi@10.42.0.160

* Restart network
sudo ifdown wlan0
sudo ifup wlan0
sudo ifconfig eth0 down
sudo ifconfig eth0 up


# =====================================
# Set up OpenCR

Tutorial website:
http://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/

export OPENCR_PORT=/dev/ttyACM0
export OPENCR_MODEL=waffle
rm -rf ./opencr_update.tar.bz2
.....

# =====================================
# Configure network
$ nano .bashrc
set the master IP as my laptop's IP
set the other one as the raspberry pi's IP

# =====================================
# Start basic ROS things on turtlebot
roslaunch turtlebot3_bringup turtlebot3_robot.launch