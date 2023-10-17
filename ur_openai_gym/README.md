# ur_openai_gym
OpenAI Gym interface for Universal Robots with ROS Gazebo based on [openai_ros](https://bitbucket.org/theconstructcore/openai_ros/src/kinetic-devel/)

### Examples
- **Reinforcement Learning** with Soft-Actor-Critic (SAC) with the implementation from [TF2RL](https://github.com/keiohta/tf2rl)
  with 2 action spaces: task-space (end-effector Cartesian space) and joint-space.
1. Start the simulation environment based on [ur3](https://github.com/cambel/ur3)
`roslaunch ur3_gazebo ur3e_cubes_example.launch` 
2. Execute the learning session:

For task-space example:
` rosrun ur_rl tf2rl_sac.py -e 0`

For task-space example:
` rosrun ur_rl tf2rl_sac.py -e 1`

## [Install with Docker](https://github.com/cambel/ur_openai_gym/wiki/Install-with-Docker)
