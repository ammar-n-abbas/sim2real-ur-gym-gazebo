# Sim2Real Implementation: Gymnasium-Gazebo UREnv for Deep Reinforcement Learning

Universal Robot Environment for OpenAI Gymnasium and ROS Gazebo Interface based on: 
[openai_ros](https://bitbucket.org/theconstructcore/openai_ros/src/kinetic-devel/), 
[ur_openai_gym](https://github.com/cambel/ur_openai_gym),
[rg2_simulation](https://github.com/ekorudiawan/rg2_simulation), and
[gazeboo_grasp_fix_plugin](https://github.com/JenniferBuehler/gazebo-pkgs/wiki/The-Gazebo-grasp-fix-plugin)

## Table of Contents

- [About](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project

Using Deep reinforcement Learning for robotics case study with the main motivation of moving fro sim2real with safety-critical systems.




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




<!-- ROADMAP -->
## Roadmap

- [x] Add basic infrastructure
- [x] Add URReachEnv
- [x] Add URGraspEnv
- [x] Add URPickAndPlaceEnv




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

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
This project is part of the research activities done along Collaborative Intelligence for Safety-Critical systems (CISC) project that has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under the Marie Skłodowska-Curie grant agreement no. 955901. (cicproject.eu)

