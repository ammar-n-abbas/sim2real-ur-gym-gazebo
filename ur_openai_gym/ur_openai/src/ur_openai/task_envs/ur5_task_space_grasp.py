#!/usr/bin/env python
import sys

sys.path.insert(0, "./src/")
sys.path.insert(0, "./src/ur5/ur_control/src/")
sys.path.insert(0, "./src/gazebo-pkgs")

import datetime
import rospy
import numpy as np
import sys
import pandas as pd
import time
import threading
import utils

from gymnasium import spaces
from gymnasium.wrappers import TimeAwareObservation

from ur_control import transformations, spalg
from ur_control.constants import DONE, FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND, SPEED_LIMIT_EXCEEDED

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur5_goal_env
from ur_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose, apply_workspace_contraints
from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.basic_models import SPHERE
from ur_gazebo.model import Model
from ur_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params

from gazebo_msgs.msg import ModelState
from gazebo_grasp_plugin_ros.msg import GazeboGraspEvent
from gazebo_msgs.srv import GetModelState, SetModelState, GetModelStateRequest
from control_msgs.msg import JointControllerState
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, WrenchStamped
from gazebo_msgs.msg import ContactsState
import tf

# import rviz_tools_py as rviz_tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5GraspEnv(ur5_goal_env.UR5Env):
    def __init__(self):
        # node_name = 'ur5_rl_zoo3_sim'
        # rospy.init_node(node_name, anonymous=True, log_level=rospy.WARN)

        load_ros_params(rospackage_name="ur_rl",
                        rel_path_from_package_to_file="config",
                        yaml_file_name="simulation/task_space_pick_and_place.yaml")

        self.cost_positive = False
        self.get_robot_params()
        self.rate = rospy.Rate(1 / self.agent_control_dt)

        np.random.seed(self.rand_seed)

        ur5_goal_env.UR5Env.__init__(self, self.driver)

        # self.get_hmm_obs = None

        self.drl_state_publisher = rospy.Publisher("/ur_gym/drl/state", String, queue_size=10)
        self.drl_action_publisher = rospy.Publisher("/ur_gym/drl/action", String, queue_size=10)
        self.drl_reward_publisher = rospy.Publisher("/ur_gym/drl/reward", String, queue_size=10)

        # self.standard_obs = StandardScaler()
        # self.minmax_rew = MinMaxScaler()
        # self.stop_thread = False
        # self.done_bool = False

        # ur5_goal_env.UR5Env.__init__(self, self.driver)

        self._previous_joints = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.vel_per_violation = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None
        self.train_steps = 0

        self.ee_position_publisher = rospy.Publisher("/ur/ee_position", String, queue_size=10)

        self.cube_pose_publisher = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)
        self.bar_dist_pose_publisher = rospy.Publisher('/bar_dist_pose', PoseStamped, queue_size=10)
        self.listener = tf.TransformListener()

        self.marker1_pub = rospy.Publisher("/visualization_marker1", Marker, queue_size=10)
        self.marker1 = Marker()
        self.marker2_pub = rospy.Publisher("/visualization_marker2", Marker, queue_size=10)
        self.marker2 = Marker()
        self.marker3_pub = rospy.Publisher("/visualization_marker3", Marker, queue_size=10)
        self.marker3 = Marker()

        self.gripper_attached = False
        self.prev_grip = False
        self.gripper_state_sub = rospy.Subscriber('/gazebo_grasp_plugin_event_republisher/grasp_events',
                                                  GazeboGraspEvent,
                                                  self.gripper_state,
                                                  queue_size=10)

        self.collision = False
        rospy.Subscriber('/imr_cell_collision', ContactsState, self.collision_callback)

        self.cube_collision = False
        rospy.Subscriber('/bumper_states', ContactsState, self.cube_collision_callback)

        self.bar_dist_collision = False
        rospy.Subscriber('/bar_dist_collision', ContactsState, self.bar_dist_collision_callback)

        self.force_magnitude = None
        rospy.Subscriber('/wrench', WrenchStamped, self.wrench_callback)

        self.last_actions = np.zeros(self.n_actions)

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array(
            [cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        bar_dist_pose = self.get_bar_dist_pose()
        bar_dist_position = np.array([bar_dist_pose.position.x, bar_dist_pose.position.y, bar_dist_pose.position.z])

        self.cube_pose_prev = cube_pose

        self.goal_a = cube_pose[:3]
        obs = self._get_obs()

        self.reward_threshold = 500.0
        self.n_count = 0

        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions,),
                                       dtype=np.float32)

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=np.float32),
        ))

        # self.replay_buffer = [utils.ReplayBuffer(self.observation_space['observation'].shape[0], 4, max_size=int(1e6)) for i in
        #                       range(4)]

        self.evaluate = False

        self.violations = []
        self.count_speed_violation = 0
        self.count_ik_violation = 0
        self.count_coll_violation = 0
        self.count_obj_coll_violation = 0
        self.count_vel_violation = 0
        self.success = 0
        self.trials = 1

        self.violation_filename = datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + "_violations.csv"

        print("ACTION SPACES TYPE", self.action_space)
        print("OBSERVATION SPACES TYPE", self.observation_space)

        # self.spawner = GazeboModels('ur_gazebo')
        # self.start_model = Model("start", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
        #                          "ball", 0.01, "RedTransparent"), reference_frame="base_link")
        # self.target_model = Model("target", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
        #                           "ball", 0.1, "GreenTransparent"), reference_frame="base_link")

    def start_expert_thread(self):
        expert_thread = threading.Thread(target=self.get_expert_thread)
        expert_thread.start()
        return expert_thread

    def stop_expert_thread(self, expert_thread):
        self.stop_thread = True
        expert_thread.join()

    def get_expert_thread(self):
        next_state = None
        while not self.stop_thread:
            obs = self._get_obs()

            reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info=None)

            cube_pose = self.get_cube_pose()
            cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)[:3]
            action = np.concatenate([np.linspace(cpose, cube_position, 1)[0], [-1]]) if not self.gripper_attached \
                else np.concatenate([np.linspace(cube_position, self.goal_b, 1)[0], [1]])

            # self.drl_action_publisher.publish(str(action))

            if next_state is None: next_state = obs['observation']
            # self.replay_buffer[0].add(obs['observation'], action, next_state, reward, self.done_bool)
            next_state = obs['observation']

            self.rate.sleep()

    def pick_and_place(self):
        self.done_bool = False
        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array(
            [cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        '''Reach'''
        self.ur5_arm.set_target_pose_flex(cube_pose, t=0.5)
        time.sleep(1)
        '''Pick'''
        self.ur5_gripper.gripper_joint_control(self.map_gripper(1.))
        rospy.sleep(1)
        '''Place'''
        self.ur5_arm.set_target_pose_flex(np.concatenate([self.goal_b, [0, 0, 0, 1]]), t=0.5)
        rospy.sleep(1)
        self.done_bool = True

    def pick(self):
        time.sleep(1)
        # self.done_bool = False
        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array(
            [cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        '''Reach'''
        self.ur5_arm.set_target_pose_flex(cube_pose, t=0.5)
        time.sleep(1)
        '''Pick'''
        self.ur5_gripper.gripper_joint_control(self.map_gripper(1.))
        time.sleep(1)
        # self.done_bool = True

    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.prev_grip = False
        self.last_actions = np.zeros(self.n_actions)
        if self.evaluate:
            self.violations.append({
                                    "steps": self.train_steps,
                                    "collision": self.count_coll_violation,
                                    "speed violation": self.count_speed_violation,
                                    "ik violation": self.count_ik_violation,
                                    "object collision": self.count_obj_coll_violation,
                                    "velocity violation": self.count_vel_violation,
                                    "velocity during collision": np.mean(self.vel_per_violation),
                                    "success": self.success,
                                    "return": np.sum(self.reward_per_step),
                                    })
            pd.DataFrame(self.violations).to_csv(self.violation_filename)
        self.count_coll_violation = 0
        self.count_speed_violation = 0
        self.count_ik_violation = 0
        self.count_obj_coll_violation = 0
        self.count_vel_violation = 0
        self.success = 0
        self.reward_per_step = []
        self.vel_per_violation = []

    def create_marker(self, marker, marker_pub, marker_type=2, target_location=None, color=None, scale=None):
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = marker_type
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = target_location[0]
        marker.pose.position.y = target_location[1]
        marker.pose.position.z = target_location[2]

        marker_pub.publish(marker)

    def collision_callback(self, msg):
        self.collision = False
        if len(msg.states) > 0:
            for contact in msg.states:
                if 'robot' in contact.collision1_name or 'robot' in contact.collision2_name:
                    self.collision = True

    def bar_dist_collision_callback(self, msg):
        self.bar_dist_collision = False
        if len(msg.states) > 0:
            for contact in msg.states:
                if 'robot' in contact.collision1_name or 'robot' in contact.collision2_name:
                    self.bar_dist_collision = True

    def cube_collision_callback(self, msg):
        self.cube_collision = False
        if len(msg.states) > 0:
            for contact in msg.states:
                if 'robot' in contact.collision1_name or 'robot' in contact.collision2_name:
                    self.cube_collision = True

    def wrench_callback(self, data):
        self.force_magnitude = (data.wrench.force.x ** 2 + data.wrench.force.y ** 2 + data.wrench.force.z ** 2) ** 0.5

    def set_model_state(self):
        model_names = ['cube']
        if self.object_disturbance:
            model_names = ['cube', 'bar_dist']
        for model_name in model_names:
            rospy.wait_for_service('/gazebo/set_model_state')
            self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

            state_msg = ModelState()
            state_msg.model_name = model_name
            dx = np.random.uniform(0.0, 0.2, 1).ravel() if self.random_cube_pose else 0
            dy = np.random.uniform(-0.0, -0.5, 1).ravel() if self.random_cube_pose else 0
            state_msg.pose.position.x = 0.6 + dx
            state_msg.pose.position.y = -0.4 + dy
            state_msg.pose.position.z = 1.0
            state_msg.pose.orientation.w = np.pi / 2
            # state_msg.pose.orientation.x = 0
            # state_msg.pose.orientation.y = 0
            state_msg.pose.orientation.z = -np.pi / 2

            self.set_state(state_msg)

    def get_model_state_client(self, model_name, relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        return get_model_state(model_name, relative_entity_name)

    def transform_to_base_link(self, pose):
        self.listener.waitForTransform('base_link', pose.header.frame_id, rospy.Time(), rospy.Duration(1.0))
        base_link_pose = self.listener.transformPose('base_link', pose)
        return base_link_pose

    def get_bar_dist_pose(self):
        response = self.get_model_state_client('bar_dist', 'world')
        if response:
            response = self.transform_to_base_link(response)

            transformed_pose_stamped = PoseStamped()
            transformed_pose_stamped.header.frame_id = 'base_link'
            transformed_pose_stamped.pose = response.pose
            transformed_pose_stamped.pose.position.z -= 0.1

            transformed_pose_stamped.header.stamp = rospy.Time.now()
            self.bar_dist_pose_publisher.publish(transformed_pose_stamped)

            return transformed_pose_stamped.pose

    def get_cube_pose(self):
        response = self.get_model_state_client('cube', 'world')
        if response:
            response = self.transform_to_base_link(response)

            transformed_pose_stamped = PoseStamped()
            transformed_pose_stamped.header.frame_id = 'base_link'
            transformed_pose_stamped.pose = response.pose
            transformed_pose_stamped.pose.position.z -= 0.1

            transformed_pose_stamped.header.stamp = rospy.Time.now()
            self.cube_pose_publisher.publish(transformed_pose_stamped)

            return transformed_pose_stamped.pose

    def _log(self):
        if self.obs_logfile is None:
            try:
                self.obs_logfile = rospy.get_param("ur5_gym/output_dir") + "/state_" + \
                                   datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
            except Exception:
                return
        print("Initiaalizing log")
        self.reward_per_step = []
        self.vel_per_violation = []
        self.obs_per_step = []

    def gripper_state(self, msg):
        self.gripper_attached = msg.attached
        # record_screen
        print("grasped") if self.gripper_attached and not self.bar_dist_collision else print("")
        return self.gripper_attached

    def map_gripper(self, grip_cmd):
        return -1.0 if np.round(grip_cmd, 3) > 0. else 0.8

    def get_robot_params(self):
        prefix = "ur_gym"
        load_param_vars(self, prefix)

        driver_param = rospy.get_param(prefix + "/driver")
        self.param_use_gazebo = True
        if driver_param == "robot":
            self.param_use_gazebo = False

        self.rand_seed = rospy.get_param(prefix + "/rand_seed", None)
        self.rand_init_interval = rospy.get_param(prefix + "/rand_init_interval", 5)
        self.rand_init_counter = 0
        self.rand_init_cpose = None

    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(
                self.ur5_arm.end_effector(np.deg2rad(self.init_q).tolist(), tip_link=self.ur5_arm.ee_link),
                self.workspace, self.reset_time)
            self.rand_init_counter = 0
        self.rand_init_counter += 1

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self._log()

        if self.random_initial_pose:
            self._randomize_initial_pose()
            self.ur5_arm.set_target_pose(pose=self.rand_init_cpose,
                                         wait=True,
                                         t=self.reset_time)
        else:
            qc = np.deg2rad(self.init_q).tolist()
            self.ur5_arm.set_joint_positions(position=qc,
                                             wait=True,
                                             t=self.reset_time)

        cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)

        if self.random_target_pose:
            if self.cl_target_pose:
                self.n_count += 1
                d = self.n_count * 0.001
                if d >= 1:
                    self.n_count = 0
                    print("Resetting Curriculum Learning")
                cube_pose = self.get_cube_pose()
                cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
                cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z,
                                             cube_pose.orientation.w])
                cube_pose = np.concatenate((cube_position, cube_orientation))

                self.goal_a = cube_pose[:3]
                self.goal_b = cube_pose[:3] + self.np_random.uniform(-d, d, size=3)
            # print("\nTarget pose:", self.goal_a)
            else:
                cube_pose = self.get_cube_pose()
                cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
                cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z,
                                             cube_pose.orientation.w])
                cube_pose = np.concatenate((cube_position, cube_orientation))

                self.goal_a = cube_pose[:3]
                # self.goal_b =  cpose[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
                dx = np.random.uniform(-0.3, 0.3, 1).ravel()
                dy = np.random.uniform(0.2, 0.5, 1).ravel()
                self.goal_b[0] = dx
                self.goal_b[1] = dy
                self.goal_b[2] = 0.15

        # self.start_model.set_pose(self.ur5_arm.end_effector()[:3])
        # self.spawner.update_model_state(self.start_model)

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array(
            [cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        self.goal_a = cube_pose[:3]

        self.create_marker(self.marker1, self.marker1_pub, marker_type=1, target_location=self.goal_a, color=[1, 0, 0],
                           scale=self.cube_size)
        self.create_marker(self.marker2, self.marker2_pub, marker_type=2, target_location=self.goal_b, color=[0, 1, 0],
                           scale=0.025)

        # self.target_model.set_pose(self.goal_a)
        self.ur5_gripper.gripper_joint_control(1.0)

    def get_cube_error_and_vels(self, cube_pose):
        cube_pos_now = cube_pose
        cube_pos_last = self.cube_pose_prev

        linear_velocity = (cube_pos_now[:3] - cube_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            cube_pos_now[3:], cube_pos_last[3:], self.agent_control_dt)
        cube_velocity = np.concatenate((linear_velocity, angular_velocity))

        cube_error = spalg.translation_rotation_error(self.goal_b, cube_pos_now)

        self.cube_pose_prev = cube_pose.copy()
        return cube_error, cube_velocity

    def get_points_and_vels(self, joint_angles):
        if self._previous_joints is None:
            self._previous_joints = self.ur5_arm.joint_angles()

        ee_pos_now = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ee_link)

        ee_pos_last = self.ur5_arm.end_effector(joint_angles=self._previous_joints, tip_link=self.ur5_arm.ee_link)
        self._previous_joints = joint_angles

        linear_velocity = (ee_pos_now[:3] - ee_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            ee_pos_now[3:], ee_pos_last[3:], self.agent_control_dt)
        velocity = np.concatenate((linear_velocity, angular_velocity))

        error = spalg.translation_rotation_error(self.goal_a, ee_pos_now)

        return error, velocity

    def _get_obs(self):
        joint_angles = self.ur5_arm.joint_angles()
        ee_pose = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ee_link)
        ee_points, ee_velocities = self.get_points_and_vels(joint_angles)

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array(
            [cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))
        self.goal_a = cube_pose[:3].copy()
        cube_error, cube_velocity = self.get_cube_error_and_vels(cube_pose)

        bar_dist_pose = self.get_bar_dist_pose()
        bar_dist_position = np.array([bar_dist_pose.position.x, bar_dist_pose.position.y, bar_dist_pose.position.z])

        if self.object_disturbance:
            obs = np.concatenate([
                # np.array(self.gripper_attached, dtype=float).ravel(),
                # ee_pose.ravel()[:3],
                bar_dist_position.ravel(),
                joint_angles.ravel(),
                # self.goal_a.ravel(),
                # self.goal_b.ravel(),
                ee_points.ravel()[:3],
                ee_velocities.ravel()[:3],
                cube_error.ravel()[:3],
                cube_velocity.ravel()[:3],
                # [self.force_magnitude],
            ], dtype=np.float32)
        else:
            obs = np.concatenate([
                # np.array(self.gripper_attached, dtype=float).ravel(),
                # ee_pose.ravel()[:3],
                joint_angles.ravel(),
                # self.goal_a.ravel(),
                # self.goal_b.ravel(),
                ee_points.ravel()[:3],
                ee_velocities.ravel()[:3],
                cube_error.ravel()[:3],
                cube_velocity.ravel()[:3],
                # [self.force_magnitude],
            ], dtype=np.float32)

        # self.standard_obs.partial_fit(obs_data)
        # obs = self.standard_obs.transform(obs_data).ravel()

        self.create_marker(self.marker1, self.marker1_pub, marker_type=1, target_location=self.goal_a, color=[1, 0, 0],
                           scale=self.cube_size)
        self.create_marker(self.marker2, self.marker2_pub, marker_type=2, target_location=self.goal_b, color=[0, 1, 0],
                           scale=0.025)

        self.rate.sleep()
        # self.drl_state_publisher.publish(str(obs))

        # self.get_hmm_obs = [
        #     self.force_magnitude,
        #     np.linalg.norm(cube_velocity.ravel()[:3]),
        #     np.linalg.norm(cube_error.ravel()[:3]),
        #     float(self.gripper_attached),
        # ]

        # if goal_distance(ee_pose[:3], self.goal_a) < self.distance_threshold and self.gripper_attached:
        #     print("changing goals:")
        #     return {
        #     'observation': obs.copy().astype(dtype=float),
        #     'achieved_goal': self.goal_a.copy().astype(dtype=float),
        #     'desired_goal': self.goal_b.copy().astype(dtype=float),
        #     }
        self.train_steps += 1
        # else:
        return {
            'observation': obs.copy().astype(dtype=np.float32),
            'achieved_goal': ee_pose[:3].ravel().astype(dtype=np.float32),
            'desired_goal': self.goal_a.copy().astype(dtype=np.float32),
        }

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        actions = np.copy(action)

        if self.n_actions == 3:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0]))  # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
            self.actions = np.copy(action)

        if self.n_actions == 4:
            # with gripper end_effector
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions[:3] * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0]))  # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
            gripper_cmd = self.map_gripper(actions[3])
            self.ur5_gripper.gripper_joint_control(gripper_cmd)
            self.actions = np.copy(action)

        if self.n_actions == 6:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions * 0.001
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)

        self.action_result = self.ur5_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)
        # if goal_distance(self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)[:3], self.goal_a) < self.distance_threshold:
        #     self.ur5_gripper.gripper_joint_control(self.map_gripper(1.))
        # if not self.gripper_attached:
        #     self.ur5_gripper.gripper_joint_control(self.map_gripper(0.))
        # get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # response = get_model_state(model_name='imr_cell', relative_entity_name='imr_cell')
        # in_collision = response.is_collided
        # if in_collision:
        #     print("Collision detected")

        # if not self.gripper_attached:
        #     gripper_cmd = self.map_gripper(-1.)
        #     self.ur5_gripper.gripper_joint_control(gripper_cmd)
        #     cube_pose = self.get_cube_pose()
        #     cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z+0.02])
        #     cube_pose = np.concatenate((cube_position, cpose[3:]))
        #     cmd = transformations.pose_euler_to_quaternion(cube_pose, [0, 0, 0, 0, 0, 0])
        #     self.ur5_arm.set_target_pose(cmd, t=self.agent_control_dt)
        #     self.rate.sleep()
        #     gripper_cmd = self.map_gripper(1.)
        #     self.ur5_gripper.gripper_joint_control(gripper_cmd)

        self.ee_position_publisher.publish(str(cpose))
        self.drl_action_publisher.publish(str(actions))
        self.rate.sleep()

        if self.collision or self.force_magnitude >= self.force_thresh:
            self.action_result = "collision"
        return self.action_result

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(goal, self.goal_b)
        if self.reward_type == 'dense':
            ee_pose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ee_link)
            r = goal_distance(achieved_goal, goal)
            # d = 0.
            g = 0.
            s_c = 0.
            ik_c = 0.
            c_c = 0.
            g_c = 0.
            c_c_c = 0.
            c_v = 0.
            c_c_v = 0.
            b_c_c = 0.
            if self.actions[3] >= 0. and not self.gripper_attached:
                g_c = self.gripper_cost
            if self.gripper_attached and not self.bar_dist_collision:
                g = self.grip_rew
                print("object grasped")
                if r <= self.proper_grasp_threshold:
                    print("object grasped properly")
                    g = self.grip_prop_rew
                self.prev_grip = True
            if self.action_result == 'speed_limit_exceeded':
                s_c = self.speed_cost
                self.count_speed_violation += 1
            if self.action_result == 'ik_not_found':
                ik_c = self.ik_cost
                self.count_ik_violation += 1
            if self.collision:
                c_c = self.collision_cost
                rospy.logwarn("collision")
                self.count_coll_violation += 1
                _, vel = self.get_points_and_vels(self.ur5_arm.joint_angles())
                self.vel_per_violation.append(np.linalg.norm(vel[:3]))
                if np.linalg.norm(vel[:3]) >= self.vel_thresh:
                    c_v = self.coll_vel_cost
                    rospy.logwarn("collision velocity exceeded")
                    self.count_vel_violation += 1
            if self.cube_collision and self.force_magnitude >= self.force_thresh and not self.gripper_attached:
                c_c_c = self.cube_collision_cost
                rospy.logwarn("cube_collision")
                self.count_obj_coll_violation += 1
                _, vel = self.get_points_and_vels(self.ur5_arm.joint_angles())
                self.vel_per_violation.append(np.linalg.norm(vel[:3]))
                if np.linalg.norm(vel[:3]) >= self.vel_thresh:
                    c_c_v = self.cube_coll_vel_cost
                    rospy.logwarn("collision velocity exceeded")
                    self.count_vel_violation += 1
            if self.bar_dist_collision:
                b_c_c = self.bar_dist_collision_cost
                rospy.logwarn("bar_dist_collision")
                self.count_obj_coll_violation += 1
                _, vel = self.get_points_and_vels(self.ur5_arm.joint_angles())
                self.vel_per_violation.append(np.linalg.norm(vel[:3]))
                if np.linalg.norm(vel[:3]) >= self.vel_thresh:
                    c_c_v = self.cube_coll_vel_cost
                    rospy.logwarn("collision velocity exceeded")
                    self.count_vel_violation += 1
            # if d < self.distance_threshold:
            #     d = -self.place_rew
            # if d <= self.distance_threshold:
            #     d = -50
            # else:
            #     d = 0
            reward = (
                     # - d
                     - r
                     + g
                     - s_c
                     - ik_c
                     - c_c
                     - c_c_c
                     - g_c
                     - c_v
                     - c_c_v
                     - b_c_c
                     ) * self.rew_scale_factor
            # reward = -r
            # self.minmax_rew.partial_fit(reward)
            # reward = self.minmax_rew.transform(reward).ravel()[0]

            self.drl_reward_publisher.publish(str(reward))
            # self.rate.sleep()
            self.reward_per_step.append(reward)
            return np.float64(reward)

        elif self.reward_type == 'sparse':
            reward = d > self.distance_threshold
            # self.drl_reward_publisher.publish(str(reward))
            return np.float64(-reward)

        elif self.reward_type == 'very_sparse':
            reward = -(d > self.distance_threshold).astype(np.float64)
            # self.drl_reward_publisher.publish(str(reward))
            return reward

    def _is_success(self, achieved_goal, desired_goal):
        ee_pose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ee_link)
        r = goal_distance(ee_pose[:3], desired_goal).copy()
        d = goal_distance(desired_goal, self.goal_b).copy()
        self._log_message = "Final distance error: " + str(np.round(d, 3)) \
                            + (' reached cube!' if r < self.distance_threshold else '') \
                            + (' cube reached final position!' if d < self.distance_threshold else '') \
                            + (' has object!' if self.gripper_attached and not self.bar_dist_collision else '') \
                            + (' success!' if r < self.distance_threshold and self.gripper_attached else '')
        self.success = int(r < self.distance_threshold and self.gripper_attached)
        return bool(r < self.distance_threshold and d < self.distance_threshold and self.gripper_attached)

    # def _is_success(self, achieved_goal, desired_goal):
    #     r = goal_distance(achieved_goal, desired_goal)
    #     self._log_message = "Final distance error: " + str(np.round(r, 3)) \
    #                         + (' success!' if r < self.distance_threshold and self.gripper_attached else '')
    #     return bool(r < self.distance_threshold) and self.gripper_attached
