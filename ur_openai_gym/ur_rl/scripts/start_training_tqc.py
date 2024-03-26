#!/usr/bin/env python

import sys

import torch

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


def main(args, results_dir, models_dir, prefix):
    # --- Init ---
    rospy.init_node('ur5_td3',
                    anonymous=True, log_level=rospy.WARN)

    ros_param_path = load_ros_params(rospackage_name="ur_rl",
                                     rel_path_from_package_to_file="config",
                                     yaml_file_name="simulation/task_space_pick_and_place.yaml")

    max_episode_steps = rospy.get_param("/ur_gym/rl/steps_per_episode", 200)
    EPISODE_LENGTH = max_episode_steps

    # remove TimeLimit
    env = load_environment(
        rospy.get_param('ur_gym/env_id'),
        max_episode_steps=max_episode_steps).unwrapped
    eval_env = load_environment(
        rospy.get_param('ur_gym/env_id'),
        max_episode_steps=max_episode_steps).unwrapped

    # env = gym.make(args.env).unwrapped
    # eval_env = gym.make(args.env).unwrapped

    setattr(eval_env, 'evaluate', True)

    env = FlattenObservation(env)
    eval_env = FlattenObservation(eval_env)

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

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
    trainer.load(f"/home/research/Desktop/ur5_drl_ammar/ws_ur_openai_ros_sim/models/03-25 15-12/_UR5PickandPlaceEnv-v0_0")

    evaluations = []
    state, _ = env.reset()
    # state = state[0]['observation'] if type(state) == tuple else state['observation']
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0
    q_publisher = rospy.Publisher("/q_value", String, queue_size=10)

    actor.train()
    # env.pick()
    for t in range(int(args.max_timesteps)):
        action = actor.select_action(state)

        # q_publisher.publish(str(critic(torch.Tensor(np.expand_dims(state, 0)).to(DEVICE), torch.Tensor(np.expand_dims(action, 0)).to(DEVICE))))
        next_state, reward, terminated, truncated, _ = env.step(action)
        # next_state = next_state[0]['observation'] if type(next_state) == tuple else next_state['observation']

        episode_timesteps += 1

        replay_buffer.add(state, action, next_state, reward, terminated)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            trainer.train(replay_buffer, args.batch_size)

        if terminated or truncated:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
            # Reset environment
            state, _ = env.reset()
            # env.pick()
            # state = state[0]['observation'] if type(state) == tuple else state['observation']

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            file_name = f"{prefix}_{args.env}_{args.seed}"
            evaluations.append(eval_policy(actor, eval_env, EPISODE_LENGTH, eval_episodes=3))
            np.save(results_dir / file_name, evaluations)
            if args.save_model: trainer.save(models_dir / file_name)

    # eval_policy(actor, eval_env, EPISODE_LENGTH, eval_episodes=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="UR5PickandPlaceEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e10, type=int)  # Max time steps to run environment
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.05, type=float)  # Target network update rate
    parser.add_argument("--log_dir", default='')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    current_datetime = str(datetime.now().strftime("%m-%d %H-%M"))

    results_dir = log_dir / 'results' / current_datetime
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = log_dir / 'models' / current_datetime
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    main(args, results_dir, models_dir, args.prefix)
