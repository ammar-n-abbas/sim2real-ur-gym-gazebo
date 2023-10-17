import datetime
import rospy
import numpy as np
import sys

from gymnasium import spaces

from ur_control import transformations, spalg

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur5_env
from ur_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose, apply_workspace_contraints

from std_msgs.msg import String


class UR5TaskSpaceEnv(ur5_env.UR5Env):
    def __init__(self):

        self.cost_positive = False
        self.get_robot_params()

        ur5_env.UR5Env.__init__(self)

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

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs.shape,
                                            dtype='float32')

        self.trials = 1
        self.rate = rospy.Rate(1/self.agent_control_dt)

        print("ACTION SPACES TYPE", (self.action_space))
        print("OBSERVATION SPACES TYPE", (self.observation_space))

        self.ee_position_publisher = rospy.Publisher("/ur/ee_position", String, queue_size=10)


    def _init_env_variables(self):
        self.step_count = 0
        self.action_result = None
        self.last_actions = np.zeros(self.n_actions)


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
        cpose = self.ur5_arm.end_effector()
        
        if self.random_target_pose:
            self.n_count += 1
            d = self.n_count * 0.001
            if d >= 1:
                self.n_count = 0
                print ("Resetting Curriculum Learning")
            self.target_pose = np.random.uniform(cpose[:3]-d, cpose[:3]+d, (3,)).tolist() + list(cpose[3:])
            print("\nTarget pose:", self.target_pose) 

        if self.curriculum_learning:
            self.dist_cl -= 0.005
            self.distance_threshold = self.dist_cl
            if self.dist_cl <= 5:      
                self.distance_threshold = 5.
            print("\nDistance threshold:", self.distance_threshold) 

        deltax = np.array([0., 0., 0., 0., 0., 0.])
        cpose = transformations.pose_euler_to_quaternion(
            self.ur5_arm.end_effector(), deltax, ee_rotation=True)
        self.ur5_arm.set_target_pose(pose=cpose,
                                     wait=True,
                                     t=self.reset_time)
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
                    

    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(self.ur5_arm.end_effector(self.init_q), self.workspace, self.reset_time)
            self.rand_init_counter = 0
        self.rand_init_counter += 1


    def _get_obs(self):
        joint_angles = self.ur5_arm.joint_angles()
        ee_points, ee_velocities = self.get_points_and_vels(joint_angles)

        obs = np.concatenate([
            joint_angles, 
            self.ur5_arm.end_effector(joint_angles=joint_angles),
            self.target_pose,
            ee_points.ravel(),
            ee_velocities.ravel()
            ], dtype=np.float32)

        return obs

    def get_points_and_vels(self, joint_angles):
        if self._previous_joints is None:
            self._previous_joints = self.ur5_arm.joint_angles()

        ee_pos_now = self.ur5_arm.end_effector(joint_angles=joint_angles)

        ee_pos_last = self.ur5_arm.end_effector(joint_angles=self._previous_joints)
        self._previous_joints = joint_angles 

        linear_velocity = (ee_pos_now[:3] - ee_pos_last[:3]) / self.agent_control_dt
        angular_velocity = transformations.angular_velocity_from_quaternions(
            ee_pos_now[3:], ee_pos_last[3:], self.agent_control_dt)
        velocity = np.concatenate((linear_velocity, angular_velocity))

        error = spalg.translation_rotation_error(self.target_pose, ee_pos_now)

        return error, velocity


    def _compute_reward(self, observations, done): 
        cdist = np.linalg.norm(self.target_pose[:3] - self.ur5_arm.end_effector()[:3])
        if done:
            return 100.
        else:
            return -cdist * 1000
            # return -1
        

    def _set_action(self, action):
        assert action.shape == (self.n_actions, )
        if np.any(np.isnan(action)):
            rospy.logerr("Invalid NAN action(s)" + str(action))
            sys.exit()
        assert np.all(action >= -1.0001) and np.all(action <= 1.0001)

        actions = np.copy(action)

        if self.n_actions == 3:
            cpose = self.ur5_arm.end_effector()
            cmd = self.ur5_arm.end_effector()
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0])) # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)

        if self.n_actions == 4:
            cpose = self.ur5_arm.end_effector()
            delta = actions * 0.001
            delta = np.concatenate((delta[:3], [0, 0], delta[3:])) # Do not change ax and ay
            cmd = transformations.pose_euler_to_quaternion(cpose, actions)
            cmd = apply_workspace_contraints(cmd, self.workspace)

        if self.n_actions == 6:
            cpose = self.ur5_arm.end_effector()
            delta = actions * 0.001
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
       
        self.ur5_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)
        self.ee_position_publisher.publish(str(cpose))
        self.rate.sleep()


    def _is_terminated(self, observations):
        true_error = spalg.translation_rotation_error(self.target_pose, self.ur5_arm.end_effector())
        true_error[:3] *= 1000
        true_error[3:] = np.rad2deg(true_error[3:])

        success = np.linalg.norm(true_error[:3], axis=-1) < self.distance_threshold
        self._log_message = "Final distance error: " + str(np.round(true_error, 3)) \
                            + "\ndistance to target: " + str(np.round(np.linalg.norm(true_error[:3]), 3)) \
                            + (' success!' if success else '')
        return bool(success)