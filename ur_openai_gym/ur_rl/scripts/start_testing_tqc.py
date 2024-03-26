#!/usr/bin/env python

import sys

sys.path.insert(0, "/home/research/Desktop/ur5_drl_ammar/ws_ur_openai_ros_sim/src/ur5/ur_control/src")
sys.path.insert(0, "ur5/ur_control/src/")

import numpy as np
import gymnasium as gym
import argparse
import os
import copy
from pathlib import Path
from datetime import datetime

from tqc import DEVICE
from tqc import structures
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy

import rospy
from std_msgs.msg import String
from ur_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params

from gymnasium.wrappers import TimeAwareObservation, NormalizeObservation, NormalizeReward, RecordEpisodeStatistics, \
    FlattenObservation

EPISODE_LENGTH = 500


def main(args, results_dir, models_dir, prefix):
    # --- Init ---
    rospy.init_node('ur5_td3',
                    anonymous=True, log_level=rospy.WARN)

    ros_param_path = load_ros_params(rospackage_name="ur_rl",
                                     rel_path_from_package_to_file="config",
                                     yaml_file_name="simulation/task_space_pick_and_place.yaml")

    max_episode_steps = rospy.get_param("/ur_gym/rl/steps_per_episode", 200)

    # remove TimeLimit
    env = load_environment(
        rospy.get_param('ur_gym/env_id'),
        max_episode_steps=max_episode_steps).unwrapped
    setattr(env, 'evaluate', True)

    env = FlattenObservation(env)

    env = RescaleAction(env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item())
    trainer.load("/home/research/Desktop/ur5_drl_ammar/safety_standard_drl/safety_drl/03-16 15-17/_UR5PickAndPlaceEnv-v0_0")

    evaluations = []
    state, _ = env.reset()
    done = False

    for t in range(int(args.max_timesteps)):
        current_datetime = str(datetime.now().strftime("%m-%d %H-%M"))
        file_name = f"{current_datetime}_{prefix}_{args.env}_{args.seed}"
        evaluations.append(eval_policy(actor, env, EPISODE_LENGTH, eval_episodes=100000))
        np.save(results_dir / file_name, evaluations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="UR5PickAndPlaceEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e10, type=int)  # Max time steps to run environment
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.05, type=float)  # Target network update rate
    parser.add_argument("--log_dir", default='.')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    results_dir = log_dir / 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = log_dir / 'models'
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    main(args, results_dir, models_dir, args.prefix)
