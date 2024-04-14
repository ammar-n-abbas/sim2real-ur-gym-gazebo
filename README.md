# Sim2Real Implementation: Gymnasium-Gazebo UREnv for Deep Reinforcement Learning With Reach, Grasp, and Pick&Place Environment with Collision Avoidance (Object or Human)


Universal Robot Environment for OpenAI Gymnasium and ROS Gazebo Interface based on: 
[openai_ros](https://bitbucket.org/theconstructcore/openai_ros/src/kinetic-devel/), 
[ur_openai_gym](https://github.com/cambel/ur_openai_gym),
[rg2_simulation](https://github.com/ekorudiawan/rg2_simulation), and
[gazeboo_grasp_fix_plugin](https://github.com/JenniferBuehler/gazebo-pkgs/wiki/The-Gazebo-grasp-fix-plugin)

<p align="center">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/pick_place_rviz.gif" alt="Pick and place policy visualization on Rviz" width="250">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/grasp_real.gif" alt="Sim2Real zero-shot transfer of grasping policy using safe-DRL" width="250"><br>
  <em>Sim2Real zero-shot transfer of trained policy using safe-DRL</em>
</p>

<!-- <p align="center">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/sim2real_robot_grasp.png" alt="Sim2Real zero-shot transfer of grasping policy using safe-DRL" width="750"><br>
  <em>Snapshot of Sim2Real zero-shot transfer of grasping policy using safe-DRL</em>
</p> -->

<p align="center">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/obst_coll_sim.gif" alt="Environment with obstacle avoidance" width="250">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/obst_coll_real_top.gif" alt="Sim2Real zero-shot transfer of grasping policy with obstacle collision avoidance using safe-DRL top view" width="250">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/obst_coll_real_side.gif" alt="Sim2Real zero-shot transfer of grasping policy with obstacle collision avoidance using safe-DRL front view" width="200"><br>
  <em>Sim2Real zero-shot transfer of grasping policy with obstacle collision avoidance policy using safe-DRL</em>
</p>

<p align="center">
<!--   <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/obstacle.jpeg" alt="Environment with obstacle avoidance" width="250" height="280"> -->
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/arm_top_view.jpeg" alt="Arm collision avoidance for human-robot collaboration top-view" width="250" height="280"> 
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/arm_front_view.jpeg" alt="Arm collision avoidance for human-robot collaboration front-view" width="250" height="280"><br>
  <em>Object or arm Collision avoidance for human-robot collaboration</em>
</p>


## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Task Space](#task-space)
  - [State](#state)
  - [Action](#action)
  - [Reward](#reward) 
- [Usage](#usage)
  - [Launch Gazebo Simulation and Spawn the UR in the World](#launch-gazebo-simulation-and-spawn-the-ur-in-the-world)
  - [UR Gym Configuration YAML File](#ur-gym-configuration-yaml-file)
    - [General Agent Parameters](#general-agent-parameters)
    - [Initial Conditions](#initial-conditions)
    - [Workspace and Initial Pose](#workspace-and-initial-pose)
    - [Object Properties](#object-properties)
    - [Anomalies](#anomalies)
    - [Validations](#validations)
    - [Target](#target)
    - [Actions Parameters](#actions-parameters)
    - [Success Parameters](#success-parameters)
    - [Penalty Threshold](#penalty-threshold)
    - [Reward Parameters](#reward-parameters)
    - [Reinforcement Learning (RL)](#reinforcement-learning-rl)
  - [Training Script for TQC (Truncated Quantile Critic) Algorithm](#training-script-for-tqc-truncated-quantile-critic-algorithm)
    - [Arguments](#arguments)
    - [Example](#example)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)



<!-- ABOUT THE PROJECT -->
## About The Project

Using Deep reinforcement Learning for a robotics case study with the main motivation of moving from sim2real with safety-critical systems.




<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
```sh
sudo apt-get install ros-noetic-desktop-full
```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo.git
   ```
2. Build package
   ```sh
   catkin build
   ```
3. Install dependencies
   ```sh
   rosdep install --from-paths src --ignore-src -r -y
   ```

## Task Space

### State

The "state" refers to the current situation or configuration of the environment that the agent is interacting with. It includes relevant information as a set of variables or features necessary for the agent to make decisions and take action. In the `UR5Env` environment, the `state` is represented as a concatenation of various components that provide information about the current state of the robotic system. The state includes:

- Disturbance position (if applicable).
- Joint angles of the UR5 robotic arm.
- End effector $-$ cube position error.
- Cube velocity.

### Action

The "action" represents the decisions or moves an agent can take within a given environment. Actions are the choices available to the agent at any given state, and the goal of the reinforcement learning algorithm is to learn a policy that maps states to optimal actions. The `action` in the `UR5Env` environment represents the desired continuous changes to the end effector position (XYZ) and gripper control (G). The 4-dimensional action vector corresponds to:
 
- X: Translation in the X-axis (resolution: 0.001m).
- Y: Translation in the Y-axis (resolution: 0.001m).
- Z: Translation in the Z-axis (resolution: 0.001m).
- G: Gripper action (binary: open or close)

### Reward

The objective of the reward function is to shape the agent's behavior to achieve desired outcomes during robotic tasks while maintaining safety standards. The reward function incorporates multiple factors, capturing aspects such as goal attainment, grasping quality, speed violations, velocity during collision violations, and collisions. It is designed to incentivize behaviors that lead to successful goal completion, and safe operation. Distinct penalties are imposed for undesired events, such as collisions or excessive speeds, while rewards are given for executing proper grasps. The modified reward function is expressed mathematically as shown in the equation below:

$$
\text{reward} = - d + g - s_c - c_c - c_{c_c} - g_c - c_v - b_{c_c} -ik_c
$$

Where,

- $d$: Distance between the end effector and the cube.
- $g$: Reward for successfully grasping the object.
- $g_c$: Penalty for failed grasp attempt.
- $s_c$: Penalty for exceeding a predefined joint speed limit.
- $ik_c$: Penalty for inverse kinematic solution failure.
- $c_c$: Penalty for collisions with the environment.
- $c_{c_c}$: Penalty for collisions with the cube.
- $b_{c_c}$: Penalty for collisions with external disturbance.
- $c_v$: Penalty for exceeding a predefined collision velocity.



<!-- USAGE EXAMPLES -->
## Usage

### Launch Gazebo Simulation and Spawn the UR in the World

```
roslaunch ur_gazebo ur5_with_gripper_bringup.launch
```

### UR Gym Configuration YAML File

This YAML file (`ur_gym.yaml`) contains configuration parameters for the UR Gym environment.

#### General Agent Parameters

- `env_id`: Specifies the OpenAI Gym environment ID.
- `driver`: Indicates the driver used, such as "gazebo".
- `reset_robot`: Determines whether to reset the robot.
- `ft_sensor`: Specifies whether to use force/torque sensors.
- `agent_control_dt`: Time step for controlling the agent.
- `reset_time`: Time allocated for resetting the environment.
- `rand_seed`: Seed for random number generation.

#### Initial Conditions

- `random_cube_pose`: Determines if the initial cube pose is randomized.
- `random_target_pose`: Determines if the initial target pose is randomized.
- `random_initial_pose`: Determines if the initial robot pose is randomized.
- `cl_target_pose`: Specifies if the target pose is controlled.
- `dist_cl`: Distance for controlling the target pose.
- `rand_init_interval`: Interval for random initialization.

#### Workspace and Initial Pose

- `workspace`: Defines the workspace bounds.
- `init_q`: Initial joint configuration.

#### Object Properties

- `cube_size`: Size of the cube.

#### Anomalies

- `object_disturbance`: Indicates whether object disturbance is enabled.
- `random_object_disturbance`: Specifies if random object disturbance is enabled.
- `hrc_disturbance`: Indicates whether human-robot collaboration disturbance is enabled.
- `random_hrc_disturbance`: Specifies if random human-robot collaboration disturbance is enabled.

#### Validations

- `sil_validation`: Specifies whether to perform SIL (Software-in-the-loop) validation.

#### Target

- `goal_a`: Position of goal A.
- `goal_b`: Position of goal B.

#### Actions Parameters

- `n_actions`: Number of actions.

#### Success Parameters

- `distance_threshold`: Threshold for successful distance.
- `proper_grasp_threshold`: Threshold for proper grasp.

#### Penalty Threshold

- `vel_thresh`: Velocity threshold.
- `force_thresh`: Force threshold.

#### Reward Parameters

- `reward_type`: Type of reward.
- `speed_cost`: Cost for speed.
- `ik_cost`: Cost for inverse kinematics.
- `collision_cost`: Cost for collision.
- `coll_vel_cost`: Cost for collision velocity.
- `cube_collision_cost`: Cost for cube collision.
- `cube_coll_vel_cost`: Cost for cube collision velocity.
- `bar_dist_collision_cost`: Cost for bar distance collision.
- `gripper_cost`: Cost for gripper.
- `grip_rew`: Reward for gripping.
- `grip_prop_rew`: Proportional reward for gripping.
- `place_rew`: Reward for placing.
- `rew_scale_factor`: Scaling factor for rewards.

#### Reinforcement Learning (RL)

- `steps_per_episode`: Number of steps per episode for RL training.

  
### Training Script for TQC (Truncated Quantile Critic) Algorithm

This script (`start_training_tqc.py`), taken from [SamsungLabs/tqc_pytorch](https://github.com/SamsungLabs/tqc_pytorch), allows you to train an agent using the TQC algorithm in various OpenAI Gym environments.


```bash
rosrun start_training_tqc.py [--env ENV] [--eval_freq EVAL_FREQ] [--max_timesteps MAX_TIMESTEPS]
                             [--seed SEED] [--n_quantiles N_QUANTILES] [--top_quantiles_to_drop_per_net TOP_QUANTILES_TO_DROP_PER_NET]
                             [--n_nets N_NETS] [--batch_size BATCH_SIZE] [--discount DISCOUNT] [--tau TAU] [--log_dir LOG_DIR]
                             [--prefix PREFIX] [--save_model]
```

#### Arguments

- `--env`: (Default: "UR5PickandPlaceEnv-v0") The name of the OpenAI Gym environment.
- `--eval_freq`: (Default: 5000) Frequency (in time steps) at which to evaluate the agent's performance.
- `--max_timesteps`: (Default: 10000000000) Maximum number of time steps to run the environment.
- `--seed`: (Default: 0) Seed for random number generation.
- `--n_quantiles`: (Default: 25) Number of quantile samples to draw for each state-action pair.
- `--top_quantiles_to_drop_per_net`: (Default: 2) Number of top quantiles to drop per network.
- `--n_nets`: (Default: 5) Number of critic networks to use.
- `--batch_size`: (Default: 256) Batch size for both the actor and critic networks.
- `--discount`: (Default: 0.99) Discount factor for future rewards.
- `--tau`: (Default: 0.005) Rate at which to update target networks.
- `--log_dir`: (Default: "") Directory to save logs and trained models.
- `--prefix`: (Default: "") Prefix to add to log and model filenames.
- `--save_model`: (Default: True) Flag to save trained model and optimizer parameters.

#### Example

```bash
rosrun start_training_tqc.py --env UR5PickandPlaceEnv-v0 --eval_freq 10000 --max_timesteps 2000000 --seed 42 --save_model
```

This command runs the training script using the "UR5PickandPlaceEnv-v0" environment, evaluating the agent's performance every 10,000 time steps, training for a maximum of 2,000,000 time steps, with a random seed of 42, and saving the trained model and optimizer parameters.


<!-- ROADMAP -->
## Roadmap

- [x] Add basic infrastructure
- [x] Add URReachEnv
- [x] Add URGraspEnv
- [x] Add URPickAndPlaceEnv
- [x] Add object disturbance scenario
- [x] Add human arm disturbance scenario




<!-- CONTRIBUTING -->
## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.




<!-- CONTACT -->
## Contact

Ammar Abbas - ammar.abbas@scch.at




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project is part of the research activities done along the Collaborative Intelligence for Safety-Critical systems (CISC) project that has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under the Marie Skłodowska-Curie grant agreement no. 955901. (cicproject.eu)

