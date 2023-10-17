# The MIT License (MIT)
#
# Copyright (c) 2018-2022 Cristian C Beltran-Hernandez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian C Beltran-Hernandez
import sys
sys.path.insert(0, "/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_control/src")

from ur_openai import robot_env
import rospy
import numpy as np
from numpy.random import RandomState

from ur_control.arm import Arm
from ur_openai import robot_env

class UR5Env(robot_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """

        rospy.logdebug("Start UREnv Init")
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['joint_state_controller','joint_trajectory_controller']

        # It doesn't use namespace
        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_env.RobotGazeboEnv
        super(UR5Env, self).__init__(controllers_list=self.controllers_list,
                                    robot_name_space=self.robot_name_space,
                                    reset_controls=reset_controls_bool,
                                    use_gazebo=self.param_use_gazebo,
                                    start_init_physics_parameters=True)
        self.robot_connection.unpause()

        rospy.logdebug("UREnv unpause...")

        self.ur5_arm = Arm()

        if self.rand_seed is not None:
            self.seed(self.rand_seed)
            RandomState(self.rand_seed)
            np.random.seed(self.rand_seed)

        rospy.logdebug("Finished UREnv INIT...")


    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _pause_env(self):
        current_pose = self.ur5_arm.joint_angles
        input("Press Enter to continue")
        self.ur5_arm.set_joint_positions(current_pose, wait=True, t=self.reset_time)


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialized each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_terminated(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
