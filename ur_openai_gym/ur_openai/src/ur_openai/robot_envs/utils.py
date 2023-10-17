import sys
sys.path.insert(0, "/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_gazebo/src/")

import rospy
import numpy as np
import datetime
from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.model import Model
from ur_gazebo.basic_models import PEG_BOARD, BOX
from ur_control import transformations as tr, conversions


def load_param_vars(self, prefix):
    """
        Dynamically load class variables from ros params based on a prefix
    """
    params = rospy.get_param_names()
    for param in params:
        if prefix in param:
            var_name = '_'.join(param.split('/')[2:])
            value = rospy.get_param(param)
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, var_name, value)

def verify_reward_episode(reward_per_step):
    goal_index = 0
    for i, step in reward_per_step:
        if step[5] >= 0:
            goal_index = i
            break
    return reward_per_step[:goal_index]

def save_log(obs_logfile, obs_per_step, reward_per_step, cost_ws):
    if len(obs_per_step) == 0:
        return

    if len(reward_per_step) > 0:
        reward_per_step = np.array(reward_per_step)
        reward_per_step = np.sum(reward_per_step, axis=0).flatten()

        preward = np.copy(reward_per_step)
        ws = (cost_ws/cost_ws.sum()) if cost_ws.sum() > 0 else cost_ws
        preward[:3] = preward[:3] * ws
        print("d: %s, f: %s, a: %s, r: %s, cs: %s, ik: %s, cll: %s" % tuple(np.round(preward.tolist(), 1)))

    try:
        tmp = np.load(obs_logfile, allow_pickle=True).tolist()
        tmp.append(obs_per_step)
        np.save(obs_logfile, tmp)
        tmp = None
    except IOError:
        np.save(obs_logfile, [obs_per_step], allow_pickle=True)


def apply_workspace_contraints(target_pose, workspace):
    """
        target pose: Array. [x, y, z, ax, ay, az]
        workspace: Array. [[min_x, max_x], [min_y, max_y], ..., [min_az, max_az]]

        Even if the action space is smaller, define the workspace for every dimension
    """
    if len(target_pose) == 7:
        tpose = np.concatenate([target_pose[:3], tr.euler_from_quaternion(target_pose[3:])])
        res = np.array([np.clip(target_pose[i], *workspace[i]) for i in range(6)])
        return np.concatenate([res[:3], tr.quaternion_from_euler(*res[3:])])
    else:
        return np.array([np.clip(target_pose[i], *workspace[i]) for i in range(6)])

def randomize_initial_pose(initial_pose, workspace, reset_time):
    rand = np.random.uniform(low=-1.0, high=1.0, size=6)
    rand = np.array([np.interp(rand[i], [-1., 1.], workspace[i]) for i in range(6)])
    rand[3:] = np.deg2rad(rand[3:]) / reset_time  # rough estimation of angular velocity
    return tr.pose_from_angular_velocity(initial_pose, rand, dt=reset_time, ee_rotation=True)


def create_gazebo_marker(pose, reference_frame, marker_id=None):
    marker_pose = [pose[:3].tolist(), pose[3:].tolist()]
    return Model("visual_marker", marker_pose[0], orientation=marker_pose[1], reference_frame=reference_frame, model_id=marker_id)


def get_value_from_range(value, base, range_constant, mtype='sum'):
    if mtype == 'sum':
        kp_min = base - range_constant
        kp_max = base + range_constant
        return np.interp(value, [-1, 1], [kp_min, kp_max])
    elif mtype == 'mult':
        kp_min = base / range_constant
        kp_max = base * range_constant
        if value >= 0:
            return np.interp(value, [0, 1], [base, kp_max])
        else:
            return np.interp(value, [-1, 0], [kp_min, base])


def concat_vec(x, y, z, steps):
    x = x.reshape(-1, steps)
    y = y.reshape(-1, steps)
    z = z.reshape(-1, steps)
    return np.concatenate((x, y, z), axis=0).T


def spiral(radius, theta_offset, revolutions, steps):
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return x, y


def get_conical_helix_trajectory(p1, p2, steps, revolutions=5.0):
    """ Compute Cartesian conical helix between 2 points"""
    euclidean_dist = np.linalg.norm(np.array(p2[:2])-np.array(p1[:2]))
    radius = np.linspace(euclidean_dist, 0, steps)
    theta_offset = np.arctan2((p1[1] - p2[1]), (p1[0]-p2[0]))

    x, y = spiral(radius, theta_offset, revolutions, steps)
    x += p2[0]
    y += p2[1]
    z = np.linspace(p1[2]-(p1[2]-p2[2])/2, p2[2], steps)
    return concat_vec(x, y, z, steps)
