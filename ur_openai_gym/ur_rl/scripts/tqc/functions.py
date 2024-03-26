import torch

from tqc import DEVICE
import numpy as np


def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    for _ in range(0, eval_episodes):
        state, _ = eval_env.reset()
        done = False
        # state = state[0]['observation'] if type(state) == tuple else state['observation']
        t = 0
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = True if truncated or terminated else False
            avg_reward += reward
            t += 1
    avg_reward /= eval_episodes
    policy.train()
    return avg_reward


def eval_policy_srla(policy, eval_env, max_episode_steps, eval_episodes=10, n_agents=2):
    for i in range(n_agents):
        policy[i].eval()
    avg_reward = 0.
    prev_state = 0
    for _ in range(0, eval_episodes):
        if np.linalg.norm(eval_env.ur5_arm.end_effector(tip_link=eval_env.ur5_arm.ee_link)[:3] -
                          eval_env.goal_a, axis=-1) < eval_env.distance_threshold:  # pick
            i = 0
            if eval_env.gripper_attached:  # place
                i = 1
        else:
            i = 0  # reach
        if i != prev_state: print("Hidden State: ", i)
        prev_state = i
        state, _ = eval_env.reset()
        done = False
        # state = state[0]['observation'] if type(state) == tuple else state['observation']
        t = 0
        while not done and t < max_episode_steps:
            action = policy[i].select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = True if truncated or terminated else False
            avg_reward += reward
            t += 1
    avg_reward /= eval_episodes
    for i in range(n_agents):
        policy[i].train()
    return avg_reward


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
