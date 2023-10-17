import datetime
import rospy
import numpy as np
import sys

from gymnasium import spaces

from ur_control import transformations, spalg

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur5_goal_env
from ur_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose, apply_workspace_contraints

from std_msgs.msg import String
from visualization_msgs.msg import Marker

# import rviz_tools_py as rviz_tools


def goal_distance(goal_a, goal_b):
        return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5ReachEnv(ur5_goal_env.UR5Env):
    def __init__(self):

        self.cost_positive = False
        self.get_robot_params()

        ur5_goal_env.UR5Env.__init__(self, self.driver)

        self._previous_joints = None
        self.obs_logfile = None
        self.reward_per_step = []
        self.obs_per_step = []
        self.max_dist = None
        self.action_result = None

        self.last_actions = np.zeros(self.n_actions)
        obs = self._get_obs()

        self.reward_threshold = 500.0
        self.n_count = 0

        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.n_actions, ),
                                       dtype='float32')

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.trials = 1
        self.rate = rospy.Rate(1/self.agent_control_dt)

        print("ACTION SPACES TYPE", (self.action_space))
        print("OBSERVATION SPACES TYPE", (self.observation_space))

        self.ee_position_publisher = rospy.Publisher("/ur/ee_position", String, queue_size=10)

        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 10)
        self.marker = Marker()


    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)


    def create_marker(self):
        self.marker.header.frame_id = "base_link"
        self.marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        self.marker.type = 2
        self.marker.id = 0

        # Set the scale of the marker
        self.marker.scale.x = 0.025
        self.marker.scale.y = 0.025
        self.marker.scale.z = 0.025

        # Set the color
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

        # Set the pose of the marker
        self.marker.pose.position.x = self.goal_a[0]
        self.marker.pose.position.y = self.goal_a[1]
        self.marker.pose.position.z = self.goal_a[2]
        self.marker_pub.publish(self.marker)


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


    def get_robot_params(self):
        prefix = "ur_gym"         
        load_param_vars(self, prefix)

        driver_param = rospy.get_param(prefix + "/driver")
        self.param_use_gazebo = False
        if driver_param == "robot":
            self.param_use_gazebo = False
       
        self.rand_seed = rospy.get_param(prefix + "/rand_seed", None)
        self.rand_init_interval = rospy.get_param(prefix + "/rand_init_interval", 5)
        self.rand_init_counter = 0
        self.rand_init_cpose = None


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
            qc = self.init_q
            self.ur5_arm.set_joint_positions(position=qc,
                                             wait=True,
                                             t=self.reset_time)
        
        cpose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ft_frame)
        
        if self.random_target_pose:
            if self.cl_target_pose:
                self.n_count += 1
                d = self.n_count * 0.001
                if d >= 1:
                    self.n_count = 0
                    print ("Resetting Curriculum Learning")
                self.goal_a = np.random.uniform(cpose[:3]-d, cpose[:3]+d, (3,))
            # print("\nTarget pose:", self.goal_a)          
            else: 
                self.goal_a = np.random.uniform(cpose[:3]-0.15, cpose[:3]+0.15, (3,))

        self.create_marker()


    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(self.ur5_arm.end_effector(self.init_q, tip_link=self.ur5_arm.ft_frame), self.workspace, self.reset_time)
            self.rand_init_counter = 0
        self.rand_init_counter += 1


    def get_points_and_vels(self, joint_angles):
        if self._previous_joints is None:
            self._previous_joints = self.ur5_arm.joint_angles()

        ee_pos_now = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ft_frame)

        ee_pos_last = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ft_frame)
        self._previous_joints = joint_angles 

        linear_velocity = (ee_pos_now[:3] - ee_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            ee_pos_now[3:], ee_pos_last[3:], self.agent_control_dt)
        velocity = np.concatenate((linear_velocity, angular_velocity))

        error = spalg.translation_rotation_error(self.goal_a, ee_pos_now)

        return error, velocity
    
    def _get_obs(self):
        joint_angles = self.ur5_arm.joint_angles()
        ee_points, ee_velocities = self.get_points_and_vels(joint_angles)

        obs = np.concatenate([
            # joint_angles, 
            self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ft_frame)[:3],
            # self.goal_a[:3],
            # ee_points.ravel()[:3],
            ee_velocities.ravel()[:3]
            ], dtype=np.float32)
        
        achieved_goal = self.ur5_arm.end_effector(joint_angles=joint_angles, tip_link=self.ur5_arm.ft_frame).copy().ravel()[:3]

        return {
        'observation': obs.copy().astype(dtype=np.float32),
        'achieved_goal': achieved_goal.copy().astype(dtype=np.float32),
        'desired_goal': self.goal_a.copy().astype(dtype=np.float32),
        }
    

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'very_sparse':
            if d < self.distance_threshold:
                return 1.
            else:
                return 0.
        elif self.reward_type == 'dense':
            return -d
             

    def _set_action(self, action):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        actions = np.copy(action)

        if self.n_actions == 3:
            cpose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ft_frame)
            cmd = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ft_frame)
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0])) # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)

        if self.n_actions == 4:
            cpose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ft_frame)
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0], delta[3:])) # Do not change ax and ay
            cmd = transformations.pose_euler_to_quaternion(cpose, actions)
            cmd = apply_workspace_contraints(cmd, self.workspace)

        if self.n_actions == 6:
            cpose = self.ur5_arm.end_effector(tip_link=self.ur5_arm.ft_frame)
            delta = actions * 0.001
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
       
        self.ur5_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)
        self.ee_position_publisher.publish(str(cpose))
        self.rate.sleep()


    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        self._log_message = "Final distance error: " + str(np.round(d, 3)) \
                            + (' success!' if d < self.distance_threshold else '')
        return bool(d < self.distance_threshold)