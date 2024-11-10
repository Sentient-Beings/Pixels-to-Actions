# 🚀 Algorithms in Action!

In this repo, you’ll not only learn **how** different algorithms work, but you’ll also get hands-on with practical implementations in **ROS2**. My goal is to make you feel confident when working with these algorithms in your own projects. 💡

For each algorithm, you'll find a section explaining how to run and test it. Plus, there are videos showcasing each algorithm in action! 🎥 Currently, I’ve included a demo for **Particle Filtering**, but stay tuned—I'll soon be adding demos for the **Naive Kalman Filter**, **EKF**, **UKF**, and more!

Got stuck or found a bug? Feel free to open an issue! We're in this together. 🙌

## 🧪 Test Particle Filtering

### Launch the TurtleBot3 House Simulation:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```
No need to do the mapping—I've already done that for the TurtleBot3 House and provided the map. However, if you want to test AMCL in a different environment, you’ll need to do the mapping yourself and store the map in the maps folder inside the TurtleBot3 localization package.

### To generate a new map:
```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py
ros2 launch turtlebot3_cartographer cartographer_rviz.launch.py
```
This command will save the map in your current directory:
```bash
ros2 run nav2_map_server map_saver_cli -f ./map
```
### Testing AMCL with your map:
```bash
ros2 launch turtlebot3_localization nav2_amcl.launch.py
```
This will launch the AMCL node along with the map server and RViz.

## 🎬 Cool Demos for AMCL

### Cartographer Mapping Video:
[Watch the Cartographer Mapping ](https://drive.google.com/file/d/1isxhJCdbXPcoUlg7lP8My6Yyl4TPG2s3/view?usp=sharing)

### AMCL Node in Action:
[Watch the AMCL Estimating the Pose](https://drive.google.com/file/d/1tEZa_xPYZTp8tqLAbAWiHp0c97ERDbcT/view?usp=sharing)



After generating the map and running the AMCL node, you’ll see the robot’s position indicated by the red arrow in RViz. This arrow is the estimated position provided by the AMCL node. If you've read the docs, you already know how the algorithm calculates this position. 🚗💨

The purple cloud around the robot represents the uncertainty in its pose. If the robot moves away from known parts of the map, you’ll see this uncertainty grow. The same happens when you add dynamic obstacles—this demonstrates one of the limitations of the Particle Filtering approach in robot localization. 🌀

Enjoy experimenting with the algorithm, and feel free to tweak things and see how it reacts! Happy coding! 🎉