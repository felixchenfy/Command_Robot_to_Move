


# =========================================================
# SSH 

## errors and solutions
Errors: connection refused, connection closed, etc.

Good solution: $ sudo apt-get remove openssh-server, and install again. 
(Not so good) Enable for only current boot: $ sudo service ssh restart.  
(Better) Auto enable ssh at booting: $ sudo systemctl enable ssh.  

Bad solution:
$ sudo ufw allow 22 # not working !!!



# =========================================================
# VNC

## M0 (WORK)
https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-vnc-on-ubuntu-18-04
sudo apt-get install tightvncserver

Start it manually (for example by running
$ vncserver :1

Next, make the system aware of the new unit file.
$ sudo systemctl daemon-reload

Enable the unit file.
$ sudo systemctl enable vncserver@1.service

The 1 following the @ sign signifies which display number the service should appear over, in this case the default :1 as was discussed in Step 2..

Stop the current instance of the VNC server if it's still running.
$ vncserver -kill :1

Then start it as you would start any other systemd service.
$ sudo systemctl start vncserver@1

You can verify that it started with this command:
$ sudo systemctl status vncserver@1

Configure vnc as a system service. Then, run the following command to start a vnc.
$ sudo systemctl start vncserver@1

## M1 (GOOD)
https://linuxconfig.org/vnc-server-on-ubuntu-18-04-bionic-beaver-linux
password: turtle
vncviewer pi-cvlab:1

## M2 (GOOD)
https://linuxconfig.org/ubuntu-remote-desktop-18-04-bionic-beaver-linux

## M3 (BAD)
https://www.cyberciti.biz/faq/install-and-configure-tigervnc-server-on-ubuntu-18-04/
$ sudo apt install tigervnc-standalone-server tigervnc-xorg-extension tigervnc-viewer
$ vncpasswd
$ ls -l ~/.vnc/
$ vi ~/.vnc/xstartup

Append the following:
```
#!/bin/sh
# Start Gnome 3 Desktop 
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
vncconfig -iconic &
dbus-launch --exit-with-session gnome-session &
```

Start:
$ vncserver

Stop: To kill a VNC server running at desktop at :1
$ vncserver -kill :1
$ vncserver -kill :*
$ vncserver -list

## M4 (BAD)
http://simostro.synology.me/simone/2018/02/09/installing-a-vnc-server-on-linux-ubuntu-mate/
E: Unable to locate package tightvnc

## M5 (BAD)
https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-vnc-on-ubuntu-18-04
E: Unable to locate package tightvnc
sudo apt update
sudo apt install xfce4 xfce4-goodies




# =========================================================
# Real sense

https://github.com/IntelRealSense/librealsense/blob/development/doc/installation.md

How to run:
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud enable_infra1:=false enable_infra2:=false


No depth's solution page:
https://github.com/IntelRealSense/realsense-ros/issues/669


