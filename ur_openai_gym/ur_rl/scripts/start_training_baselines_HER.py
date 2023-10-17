#!/usr/bin/env python
import sys
sys.path.insert(0, "/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_control/src/")

from datetime import datetime
import glob
import os

from gymnasium.wrappers import TimeAwareObservation, NormalizeObservation, NormalizeReward
import gymnasium as gym

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from sb3_contrib import TQC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env

import numpy as np
import rospy
from std_msgs.msg import String

from ur_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params


if __name__ == '__main__':
    node_name = 'ur5_td3_HER'
    rospy.init_node(node_name,
                    anonymous=True, log_level=rospy.WARN)

    ros_param_path = load_ros_params(rospackage_name="ur_rl",
                    rel_path_from_package_to_file="config",
                    yaml_file_name="simulation/task_space_pick_and_place.yaml")

    drl_state_publisher = rospy.Publisher("/ur_gym/drl/state", String, queue_size=10)
    drl_action_publisher = rospy.Publisher("/ur_gym/drl/action", String, queue_size=10)
    drl_reward_publisher = rospy.Publisher("/ur_gym/drl/reward", String, queue_size=10)
    drl_ep_steps_publisher = rospy.Publisher("/ur_gym/drl/ep_steps", String, queue_size=10)
    drl_train_steps_publisher = rospy.Publisher("/ur_gym/drl/train_steps", String, queue_size=10)

    max_episode_steps = rospy.get_param("/ur_gym/rl/steps_per_episode", 200)
    
    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

    env = load_environment(
        rospy.get_param('ur_gym/env_id'),
        max_episode_steps=max_episode_steps)
    
    # env = gym.make('FetchPickAndPlace-v2', render_mode='human')
    
    # env = TimeAwareObservation(env)
    # env = NormalizeReward(env)
    # env = NormalizeObservation(env)

    # Initialize the model
    model = TD3("MultiInputPolicy", 
                env, 
                learning_starts=400,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=8,
                    goal_selection_strategy=goal_selection_strategy),
                verbose=1)

    list_of_files = glob.glob('./model_ckpts/*' +
                               node_name + "_" + 
                               str(rospy.get_param('ur_gym/env_id')) + "_" + 
                               str(env.reward_type) + '.zip')
    latest_file = max(list_of_files, key=os.path.getctime)
    print("loaded model file:", latest_file)
    model = model.load(latest_file, env=env)

    # model = model.load("./model_ckpts/10-10 12-37_ur5_td3_HER_dense", env=env)

    # Train and evaluate the model
    for epoch in range(int(5e100)):
        obs, info = env.reset()
        print("\nTraining:")
        model.learn(int(5e3))

        print("\nEvaluating:")
        obs, info = env.reset()
        for _ in range(int(5e2)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

            drl_ep_steps_publisher.publish(str(epoch))
            drl_state_publisher.publish(str(obs))
            drl_action_publisher.publish(str(action))
            drl_reward_publisher.publish(str(reward))
            
            if terminated or truncated:
                obs, info = env.reset()
        
        current_datetime = str(datetime.now().strftime("%m-%d %H-%M"))
        model.save("./model_ckpts/" + 
                   current_datetime + "_" + 
                   node_name + "_" + 
                   str(rospy.get_param('ur_gym/env_id')) + "_" + 
                   str(env.reward_type))
        # model.save("./model_ckpts/" + "td3_ur_env" + "_baseline")
    
