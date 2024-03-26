#!/usr/bin/env python
import sys
sys.path.insert(0, "./src/")
sys.path.insert(0, "./src/ur5/ur_control/src/")
sys.path.insert(0, "./src/gazebo-pkgs")

import datetime
import rospy
import numpy as np
import time 
from gymnasium import spaces

from ur_control import transformations, spalg

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur5_goal_env
from ur_openai.robot_envs.utils import load_param_vars, randomize_initial_pose
from ur_openai.common import load_ros_params

from std_msgs.msg import String
from ur_msgs.msg import ToolDataMsg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf
import tf2_ros
import tf2_geometry_msgs
    
    
def goal_distance(goal_a, goal_b):
        return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5PickAndPlaceRealEnv(ur5_goal_env.UR5Env):
    def __init__(self):
        node_name = 'ur5_rl_zoo3_real'
        rospy.init_node(node_name, anonymous=True, log_level=rospy.WARN)
        
        load_ros_params(rospackage_name="ur_rl",
                        rel_path_from_package_to_file="config",
                        yaml_file_name="simulation/task_space_pick_and_place_real.yaml")
                
        self.get_robot_params()

        self.drl_state_publisher = rospy.Publisher("/ur_gym/drl/state", String, queue_size=10)
        self.drl_action_publisher = rospy.Publisher("/ur_gym/drl/action", String, queue_size=10)
        self.drl_reward_publisher = rospy.Publisher("/ur_gym/drl/reward", String, queue_size=10)

        ur5_goal_env.UR5Env.__init__(self, self.driver)

        self.rate = rospy.Rate(1/self.agent_control_dt)

        self._previous_joints = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None

        self._set_init_pose()

        self.ee_position_publisher = rospy.Publisher("/ur/ee_position", String, queue_size=10)
        self.cube_pose_publisher = rospy.Publisher('/cube_pose', String, queue_size=10)
        self.listener = tf.TransformListener()
        self.listener.waitForTransform('base_link', 'aruco_marker_1', rospy.Time(), rospy.Duration(1.0))

        self.marker1_pub = rospy.Publisher("/visualization_marker1", Marker, queue_size = 10)
        self.marker1 = Marker()
        self.marker2_pub = rospy.Publisher("/visualization_marker2", Marker, queue_size = 10)
        self.marker2 = Marker()
        
        self.gripper_attached = False
        self.gripper_action = False
        self.voltage_value_prev = 0
        self.gripper_state_sub = rospy.Subscriber("/ur_hardware_interface/tool_data", ToolDataMsg, self.gripper_state)
        # self.gripper_state_open = True
        self.gripper_state_curr = np.bool(0)
        self.gripper_state_prev = np.bool(0)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.static_aruco_pub = rospy.Publisher('static_aruco_marker_pose', PoseStamped, queue_size=10)
        rospy.Subscriber('/DetectMarkers/aruco_detected_success', String, self.aruco_success_callback)
        self.aruco_detected = False

        self.cube_pose_updated = False  # Add a flag to indicate if the data is updated
        rospy.Subscriber('/DetectMarkers/aruco_detected', TransformStamped, self.get_cube_pose)
        self.last_actions = np.zeros(self.n_actions)
        while not self.cube_pose_updated:
            self.cube_pose_prev = self.cube_pose_base_link
            self.goal_a = self.cube_pose_base_link[:3] 
        
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype=float),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=float),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype=float),
        ))

        print("ACTION SPACES TYPE", (self.action_space))
        print("OBSERVATION SPACES TYPE", (self.observation_space))
        

    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)


    def create_marker(self, marker, marker_pub, type=2, target_location=None, color=None, scale=None):
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = type
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


    def aruco_success_callback(self, msg):
        if msg.data == "True":
            self.aruco_detected = True
        elif msg.data == "False":
            self.aruco_detected = False


    def get_cube_pose(self, msg):
        pose_msg = PoseStamped()

        # if msg.child_frame_id == 'aruco_marker_1':
        #     trans, rot = self.listener.lookupTransform('base_link', 'aruco_marker_1', rospy.Time())
        # else:
        trans_to_aruco, rot_to_aruco = self.listener.lookupTransform('cube_center', msg.child_frame_id, rospy.Time())
        trans, rot = self.listener.lookupTransform('base_link', 'aruco_marker_1', rospy.Time())
        trans = np.array(trans) + np.array(trans_to_aruco)
        rot = np.array(rot) + np.array(rot_to_aruco)
        # trans[2] -= 0.015

        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = trans[0]
        pose_msg.pose.position.y = trans[1]
        pose_msg.pose.position.z = trans[2]
        pose_msg.pose.orientation.x = rot[0]
        pose_msg.pose.orientation.y = rot[1]
        pose_msg.pose.orientation.z = rot[2]
        pose_msg.pose.orientation.w = rot[3]

        self.cube_pose_base_link = np.concatenate([trans, rot])
        self.cube_pose_updated = True  # Set the flag to indicate that the data is updated
        self.static_aruco_pub.publish(pose_msg) 
        self.cube_pose_publisher.publish(self.cube_pose_base_link)


    def _log(self):
        if self.obs_logfile is None:
            try:
                self.obs_logfile = rospy.get_param("ur5_gym/output_dir") + "/state_" + \
                    datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.npy'
                print("obs_logfile", self.obs_logfile)
            except Exception:
                return
        self.reward_per_step = []
        self.obs_per_step = []


    def gripper_state(self, msg):
        voltage_value = msg.analog_input2
        if np.abs(self.voltage_value_prev - voltage_value) < 0.000001 and not self.gripper_action:
            # self.gripper_state_open = True if voltage_value < 0.2 else False
            # if not self.gripper_state_open:
            self.gripper_attached = True if 1.9 < voltage_value < 2.4 else False
            print("object grasped") if self.gripper_attached else None
        self.voltage_value_prev = voltage_value
        self.rate.sleep()
        return self.gripper_attached


    def scale_gripper(self, grip_cmd):
        self.gripper_state_curr = 0 if np.round(grip_cmd, 3) > 0. else 1
        return self.gripper_state_curr 
    

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
            self.rand_init_cpose = randomize_initial_pose(self.ur5_arm.end_effector(np.deg2rad(self.init_q).tolist(), tip_link=self.ur5_arm.ee_link), self.workspace, self.reset_time)
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
            qc = np.deg2rad(self.init_q.tolist())
            # self.ur5_arm.set_joint_positions(position=qc,
            #                                  wait=True,
            #                                  t=self.reset_time)
        
        cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
        
        if self.random_target_pose:
            if self.cl_target_pose:
                self.n_count += 1
                d = self.n_count * 0.001
                if d >= 1:
                    self.n_count = 0
                    print ("Resetting Curriculum Learning")

                self.goal_a = self.cube_pose_base_link[:3]   
                self.goal_b = cpose[:3] + self.np_random.uniform(-d, d, size=3)
            # print("\nTarget pose:", self.goal_a)          
            else: 
                self.goal_a = self.cube_pose_base_link[:3]                
                self.goal_b =  cpose[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        self.goal_a = self.cube_pose_base_link[:3]

        self.ur5_gripper.gripper_joint_control(self.scale_gripper(-1))
           

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

        ee_pos_last = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ee_link)
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

        self.goal_a = self.cube_pose_base_link[:3].copy()
        cube_error, cube_velocity = self.get_cube_error_and_vels(self.cube_pose_base_link)

        obs = np.concatenate([
            np.array(float(self.gripper_attached), dtype=float).ravel(),
            ee_pose.ravel()[:3],
            joint_angles.ravel(), 
            self.goal_a.ravel(),
            self.goal_b.ravel(),
            ee_points.ravel()[:3],
            ee_velocities.ravel()[:3],
            cube_error.ravel()[:3],
            cube_velocity.ravel()[:3],
            ], dtype=float)
    
        self.create_marker(self.marker1, self.marker1_pub, type=1, target_location=self.goal_a, color=[1, 0, 0], scale=0.05)
        self.create_marker(self.marker2, self.marker2_pub, type=2, target_location=self.goal_b, color=[0, 1, 0], scale=0.025)

        self.drl_state_publisher.publish(str(obs))

        return {
        'observation': obs.copy().astype(dtype=float),
        'achieved_goal': self.goal_a.copy().astype(dtype=float),
        'desired_goal': self.goal_b.copy().astype(dtype=float),
        }
    

    def _set_action(self, action):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        actions = np.copy(action)

        if self.n_actions == 3:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            cmd = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0])) # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)

        if self.n_actions == 4:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions[:3] * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0])) # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
            gripper_cmd = self.scale_gripper(actions[3])
            if self.gripper_state_curr is not self.gripper_state_prev:
                self.gripper_action = True
                print('waiting for gripper action to complete')
                time.sleep(3.0)
                self.gripper_action = False
                self.gripper_state_prev = self.gripper_state_curr
                # self.ur5_gripper.gripper_joint_control(gripper_cmd)

        if self.n_actions == 6:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions * 0.001
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
       
        # self.action_result = self.ur5_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)
        self.ee_position_publisher.publish(str(cpose))
        self.drl_action_publisher.publish(str(actions))
        self.rate.sleep()


    def compute_reward(self, achieved_goal, goal, info):
        ee_pose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ee_link)
        r = goal_distance(ee_pose[:3], achieved_goal).copy()
        d = goal_distance(achieved_goal, goal).copy()
        g = 0.
        s_c = 0.
        ik_c = 0.
        c_c = 0.
        if self.reward_type == 'dense':
            if self.gripper_attached:
                g = self.grip_rew
                print("object grasped")
                if r <= self.proper_grasp_threshold:
                    print("object grasped properly")
                    g = self.grip_prop_rew
            if self.action_result == 'speed_limit_exceeded':
                s_c = self.speed_cost
            elif self.action_result == 'ik_not_found':
                ik_c = self.ik_cost
            elif self.action_result == 'force_exceeded':
                c_c = self.collision_cost
            reward = -r + -d + g + -s_c + -ik_c + -c_c
            return reward.astype(float)
        
        elif self.reward_type == 'sparse':
            reward = d > self.distance_thresholds
            return -reward.astype(float)
        
        elif self.reward_type == 'very_sparse':
            if d < self.distance_threshold:
                reward = 1.
                return reward
            else:
                reward = 0.
                return reward
    
        self.drl_reward_publisher.publish(str(reward))


    def _is_success(self, achieved_goal, desired_goal):
        ee_pose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ee_link)
        r = goal_distance(ee_pose[:3], achieved_goal).copy()
        d = goal_distance(achieved_goal, desired_goal).copy()
        self._log_message = "Final distance error: " + str(np.round(d, 3)) \
                            + (' reached cube!' if r < self.distance_threshold else '') \
                            + (' cube reached final position!' if d < self.distance_threshold else '') \
                            + (' has object!' if self.gripper_attached else '') \
                            + (' success!' if d < self.distance_threshold and
                                              self.gripper_attached else '')
        return bool(r < self.distance_threshold and d < self.distance_threshold and self.gripper_attached)