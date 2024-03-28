# Sim2Real Implementation: Gymnasium-Gazebo UREnv for Deep Reinforcement Learning With Reach, Grasp, and Pick&Place Environment with Collision Avoidance (Object or Human)


Universal Robot Environment for OpenAI Gymnasium and ROS Gazebo Interface based on: 
[openai_ros](https://bitbucket.org/theconstructcore/openai_ros/src/kinetic-devel/), 
[ur_openai_gym](https://github.com/cambel/ur_openai_gym),
[rg2_simulation](https://github.com/ekorudiawan/rg2_simulation), and
[gazeboo_grasp_fix_plugin](https://github.com/JenniferBuehler/gazebo-pkgs/wiki/The-Gazebo-grasp-fix-plugin)

<p align="center">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/sim2real_robot_grasp.png" alt="Sim2Real zero-shot transfer of grasping policy using safe-DRL" width="750"><br>
  <em>Sim2Real zero-shot transfer of grasping policy using safe-DRL</em>
</p>

<p align="center">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/obstacle.jpeg" alt="Environment with obstacle avoidance" width="250" height="280">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/arm_top_view.jpeg" alt="Arm collision avoidance for human-robot collaboration top-view" width="250" height="280">
  <img src="https://github.com/ammar-n-abbas/sim2real-ur-gym-gazebo/blob/master/assets/arm_front_view.jpeg" alt="Arm collision avoidance for human-robot collaboration front-view" width="250" height="280"><br>
  <em>Object or arm Collision avoidance for human-robot collaboration</em>
</p>


## Table of Contents

- [About](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


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


<!-- USAGE EXAMPLES -->
## Usage

```
roslaunch ur_gazebo ur5_with_gripper_bringup.launch
```


### Training Script for TQC (Top Quantile Critic) Algorithm

This script (`start_training_tqc.py`) allows you to train an agent using the TQC algorithm in various OpenAI Gym environments.


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

