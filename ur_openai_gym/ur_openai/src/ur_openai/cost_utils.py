import numpy as np
from ur_control.constants import DONE, FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND, SPEED_LIMIT_EXCEEDED
from ur_control import spalg


def sparse(self, done):
    if ground_true(self):
        return self.cost_goal
    return 0


def distance(self, obs, norm='standard'):
    pose = obs[:6] / self.max_distance
    distance_norm = None

    if norm == 'standard':
        distance_norm = np.linalg.norm(pose, axis=-1)
    elif norm == 'l1l2':
        distance_norm = l1l2(self, pose)

    if self.cost_positive:
        return max_range(distance_norm, 30, -70)

    return np.interp(distance_norm, [30, -70], [0, 1])


def l1l2(self, dist, weights=[1, 1, 1, 1, 1, 1.]):
    l1 = self.cost_l1 * np.array(weights)
    l2 = self.cost_l2 * np.array(weights)
    dist = dist
    norm = (0.5 * (dist ** 2) * l2 +
            np.log(self.cost_alpha + (dist ** 2)) * l1)
    return norm.sum()


def weighted_norm(vector, weights=None):
    if weights is None:
        return np.linalg.norm(vector)
    else:
        return np.linalg.norm(vector*weights)


def contact_force(self, obs):
    # If multiple reading, compute average force before applying l1l2 norm
    wrench_size = self.wrench_hist_size*6
    force = np.reshape(obs[-wrench_size:], (6,-1))
    force = np.average(force, axis=1)
    net_force = l1l2(self, force, weights=[0.35, 0.35, 0.35, 0.1, 0.1, 0.1]) #TODO fix hardcoded param
    return max_range(net_force, 10, -15) if self.cost_positive else np.interp(net_force, [10, -15], [0, 1])


def actions(self):
    last_actions = self.last_actions[:6]
    reward = np.sum(np.sqrt(last_actions**2))
    return max_range(reward, 6) if self.cost_positive else np.interp(reward, [0, self.max_action], [0, 1])


def distance_force_action_step_goal(self, obs, done, norm='l1l2'):
    cdistance = distance(self, obs, norm)
    cforce = contact_force(self, obs)
    cactions = actions(self)

    cost_ws = self.cost_ws / self.cost_ws.sum() if self.cost_ws.sum() != 0 else self.cost_ws

    reward = (np.dot(np.array([cdistance, cforce, cactions]), cost_ws))
    reward = reward * 200. / float(self.steps_per_episode)

    speed_cost = 0
    ik_cost = 0
    collision_cost = 0

    done_reward = self.cost_goal + 100*(1-self.step_count/float(self.steps_per_episode)) if ground_true(self) else 0

    if self.action_result == SPEED_LIMIT_EXCEEDED:
        speed_cost += self.cost_speed_violation

    elif self.action_result == IK_NOT_FOUND:
        ik_cost += self.cost_ik_violation

    elif self.action_result == FORCE_TORQUE_EXCEEDED:
        collision_cost += self.cost_collision

    reward += ik_cost + speed_cost + collision_cost + done_reward

    ### logging ###
    # distance, action, force, step cost, goal, position, contact force
    self.reward_per_step.append([cdistance, cforce, cactions, done_reward, speed_cost, ik_cost, collision_cost])
    return reward


def ground_true(self):
    error = spalg.translation_rotation_error(self.target_pose, self.ur5_arm.end_effector())*1000.0
    return np.linalg.norm(error[:3], axis=-1) < self.distance_threshold


def max_range(value, max_value, min_value=0):
    return np.interp(value, [min_value, max_value], [1, 0])
