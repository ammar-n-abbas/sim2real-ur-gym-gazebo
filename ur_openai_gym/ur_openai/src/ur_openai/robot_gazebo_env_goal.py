#!/usr/bin/env python3

import gymnasium_robotics as gym
from gymnasium.utils import seeding

import rospy
from .gazebo_connection import GazeboConnection, RobotConnection
from .controllers_connection import ControllersConnection
from ur_openai.msg import RLExperimentInfo

import ur_openai.log as utils
color_log = utils.TextColors()


class RobotGazeboEnv(gym.GoalEnv):
    def __init__(self, 
                 robot_name_space, 
                 controllers_list, 
                 reset_controls,
                 use_gazebo,
                 **kwargs):

        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        if use_gazebo:
            self.robot_connection = GazeboConnection(start_init_physics_parameters=False,
                                                     reset_world_or_sim="WORLD", 
                                                     **kwargs)
        else:
            self.robot_connection = RobotConnection(reset_world_or_sim="WORLD")
        self.controllers_object = ControllersConnection(
            namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Set up ROS related variables
        self.episode_num = 0
        self.step_count = 0
        self.cumulated_episode_reward = 0
        self.pause = False
        self.reward_pub = rospy.Publisher('/openai/reward',
                                          RLExperimentInfo,
                                          queue_size=1)
        
        self._log_message = None
        

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """
        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        # print("Entered step")
        # print("Unpause sim")
        # self.robot_connection.unpause()
        # print("Set action")
        # print("Action:")
        # print(action)
        self._set_action(action)
        # print("Get Obs")
        obs = self._get_obs()
        # print("Is done")
        terminated = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        truncated = True if self.step_count >= 200 else False
        info = {}
        info['is_success'] = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        self.cumulated_episode_reward += reward
        self._publish_reward_topic(reward, self.episode_num)

        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        # print("Entered reset")
        self._reset_sim()
        self._update_episode()
        self._init_env_variables()
        obs = self._get_obs()
        info = {}
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs, info

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        if self._log_message is not None:
            color_log.ok("\n>> End of Episode = %s, Reward= %s, steps=%s" %
                         (self.episode_num, self.cumulated_episode_reward, self.step_count))
            color_log.warning(self._log_message)

        rospy.logdebug("PUBLISHING REWARD...")
        self._publish_reward_topic(self.cumulated_episode_reward,
                                   self.episode_num)
        rospy.logdebug("PUBLISHING REWARD...DONE=" +
                       str(self.cumulated_episode_reward) + ",EP=" +
                       str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------
    def _reset_sim(self):
        """Resets a simulation
        """
        if self.reset_controls:
            self.robot_connection.unpause()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.robot_connection.pause()
            self.robot_connection.reset()
            self.robot_connection.unpause()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.robot_connection.pause()

        else:
            self.robot_connection.unpause()

            self._check_all_systems_ready()
            self._set_init_pose()
            # self.robot_connection.reset()

            self._check_all_systems_ready()

        self.set_model_state()

        return True

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):      

        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()
