# 🤖 Pixels to Actions

In this project, the aim is to develop the pipeline to deploy and test different VLA models to control a robotic arm in MuJoCo Sim, to perform manipulation tasks.

Currently, i am not using the 'LeRobot' hugging face codebase. But I am planning to either create a separate project testing out the LeRobot or integrate it within this project.

## Key elements of the Project

- Main simulation environemnt is MuJoCo
- A ROS2 integration is also done where an in memory datastore like Redis is used to communicate Mujoco Sim with the ROS2 and Rviz2.
- The robot can also be teleoperated or controlled using a VLA model.
- The VLA inference script is inspired from the [NanoLLM](https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/vision/vla.py)
