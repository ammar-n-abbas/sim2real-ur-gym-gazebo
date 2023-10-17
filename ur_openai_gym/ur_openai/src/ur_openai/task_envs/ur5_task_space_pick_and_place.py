import datetime
import rospy
import numpy as np
import sys

from gymnasium import spaces

from ur_control import transformations, spalg

import ur_openai.cost_utils as cost
from ur_openai.robot_envs import ur5_goal_env
from ur_openai.robot_envs.utils import load_param_vars, save_log, randomize_initial_pose, apply_workspace_contraints
from ur_gazebo.gazebo_spawner import GazeboModels
from ur_gazebo.basic_models import SPHERE
from ur_gazebo.model import Model

from gazebo_msgs.msg import ModelState 
from gazebo_grasp_plugin_ros.msg import GazeboGraspEvent
from gazebo_msgs.srv import GetModelState, SetModelState
from control_msgs.msg import JointControllerState
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import tf

# import rviz_tools_py as rviz_tools


def goal_distance(goal_a, goal_b):
        return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5PickAndPlaceEnv(ur5_goal_env.UR5Env):
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

        self.ee_position_publisher = rospy.Publisher("/ur/ee_position", String, queue_size=10)

        self.cube_pose_publisher = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)
        self.listener = tf.TransformListener()

        self.marker1_pub = rospy.Publisher("/visualization_marker1", Marker, queue_size = 10)
        self.marker1 = Marker()
        self.marker2_pub = rospy.Publisher("/visualization_marker2", Marker, queue_size = 10)
        self.marker2 = Marker()
        
        self.gripper_attached = False
        self.gripper_state_sub = rospy.Subscriber('/gazebo_grasp_plugin_event_republisher/grasp_events', 
                                                  GazeboGraspEvent, 
                                                  self.gripper_state, 
                                                  queue_size=10)

        self.last_actions = np.zeros(self.n_actions)

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        self.goal_a = cube_pose[:3] 
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

        # self.spawner = GazeboModels('ur_gazebo')
        # self.start_model = Model("start", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
        #                          "ball", 0.01, "RedTransparent"), reference_frame="base_link")
        # self.target_model = Model("target", [0, 0, 0, 0, 0, 0], file_type='string', string_model=SPHERE.format(
        #                           "ball", 0.1, "GreenTransparent"), reference_frame="base_link")


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


    def set_model_state(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        state_msg = ModelState()
        state_msg.model_name = 'cube'
        dx = np.random.uniform(-0.2, 0.1, 1).ravel() if self.random_cube_pose else 0
        dy = np.random.uniform(-0.1, 0.1, 1).ravel() if self.random_cube_pose else 0
        state_msg.pose.position.x =  0.73 + dx
        state_msg.pose.position.y = -0.73 + dy
        state_msg.pose.position.z =  1.0
        # state_msg.pose.orientation.w = 1
        # state_msg.pose.orientation.x = 0
        # state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = np.pi

        self.set_state(state_msg)

    def get_model_state_client(self, model_name, relative_entity_name):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        return get_model_state(model_name, relative_entity_name)

    def transform_to_base_link(self, pose):
        self.listener.waitForTransform('base_link', pose.header.frame_id, rospy.Time(), rospy.Duration(1.0))
        base_link_pose = self.listener.transformPose('base_link', pose)
        return base_link_pose
    
    def get_cube_pose(self):
        response = self.get_model_state_client('cube', 'world')
        if response:    
            response = self.transform_to_base_link(response)

            transformed_pose_stamped = PoseStamped()
            transformed_pose_stamped.header.frame_id = 'base_link'
            transformed_pose_stamped.pose = response.pose

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
        self.reward_per_step = []
        self.obs_per_step = []


    def gripper_state(self, msg):
        self.gripper_attached = msg.attached
        return self.gripper_attached

    def scale_gripper(self, grip_cmd):
        return -0.5 if grip_cmd > 0. else 0.5
    

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


    def _randomize_initial_pose(self, override=False):
        if self.rand_init_cpose is None or self.rand_init_counter >= self.rand_init_interval or override:
            self.rand_init_cpose = randomize_initial_pose(self.ur5_arm.end_effector(self.init_q, tip_link=self.ur5_arm.ee_link), self.workspace, self.reset_time)
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
            qc = self.init_q
            self.ur5_arm.set_joint_positions(position=qc,
                                             wait=True,
                                             t=self.reset_time)
        
        cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
        gripper_cmd = self.scale_gripper(-1)
        
        if self.random_target_pose:
            if self.cl_target_pose:
                self.n_count += 1
                d = self.n_count * 0.001
                if d >= 1:
                    self.n_count = 0
                    print ("Resetting Curriculum Learning")
                cube_pose = self.get_cube_pose()
                cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
                cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
                cube_pose = np.concatenate((cube_position, cube_orientation))

                self.goal_a = cube_pose[:3]   
                self.goal_b = cpose[:3] + self.np_random.uniform(-d, d, size=3)
            # print("\nTarget pose:", self.goal_a)          
            else: 
                cube_pose = self.get_cube_pose()
                cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
                cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
                cube_pose = np.concatenate((cube_position, cube_orientation))

                self.goal_a = cube_pose[:3]                
                self.goal_b =  cpose[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        # self.start_model.set_pose(self.ur5_arm.end_effector()[:3])
        # self.spawner.update_model_state(self.start_model)

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        self.goal_a = cube_pose[:3]

        # self.target_model.set_pose(self.goal_a)
        self.ur5_gripper.gripper_joint_control(gripper_cmd)
           

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

        cube_pose = self.get_cube_pose()
        cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z])
        cube_orientation = np.array([cube_pose.orientation.x, cube_pose.orientation.y, cube_pose.orientation.z, cube_pose.orientation.w])
        cube_pose = np.concatenate((cube_position, cube_orientation))

        self.goal_a = cube_pose[:3].copy()

        obs = np.concatenate([
            np.array(int(self.gripper_attached), dtype=np.float32).ravel(),
            ee_pose.ravel()[:3],
            joint_angles, 
            self.goal_a,
            self.goal_b,
            ee_points.ravel()[:3],
            ee_velocities.ravel()[:3]
            ], dtype=np.float32)
    
        self.create_marker(self.marker1, self.marker1_pub, type=1, target_location=self.goal_a, color=[1, 0, 0], scale=0.05)
        self.create_marker(self.marker2, self.marker2_pub, type=2, target_location=self.goal_b, color=[0, 1, 0], scale=0.025)

        
        if goal_distance(self.goal_a, ee_pose[:3]) < self.distance_threshold:
            print('achieved reach task')
            return {
            'observation': obs.copy().astype(dtype=np.float32),
            'achieved_goal': self.goal_a.copy().astype(dtype=np.float32),
            'desired_goal': self.goal_b.copy().astype(dtype=np.float32),
            }
        
        else:
            return {
            'observation': obs.copy().astype(dtype=np.float32),
            'achieved_goal': ee_pose[:3].copy().astype(dtype=np.float32),
            'desired_goal': self.goal_a.copy().astype(dtype=np.float32),
            }
    

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        f = goal_distance(self.goal_a, self.goal_b)
        g = 50 if self.gripper_attached else 0
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'very_sparse':
            if d < self.distance_threshold:
                return 1.
            else:
                return 0.
        elif self.reward_type == 'dense':
            return -d + -f + g
             

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
            # with gripperend_effector
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions[:3] * 0.001
            delta = np.concatenate((delta[:3], [0, 0, 0])) # Do not change ax, ay, and az
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
            gripper_cmd = self.scale_gripper(actions[3])
            self.ur5_gripper.gripper_joint_control(gripper_cmd)

        if self.n_actions == 6:
            cpose = self.ur5_arm.end_effector(joint_angles=None, tip_link=self.ur5_arm.ee_link)
            delta = actions * 0.001
            cmd = transformations.pose_euler_to_quaternion(cpose, delta)
       
        self.ur5_arm.set_target_pose_flex(cmd, t=self.agent_control_dt)


        # gripper_cmd = self.scale_gripper(-1.)
        # self.ur5_gripper.gripper_joint_control(gripper_cmd)
        # cube_pose = self.get_cube_pose()
        # cube_position = np.array([cube_pose.position.x, cube_pose.position.y, cube_pose.position.z+0.02])
        # cube_pose = np.concatenate((cube_position, cpose[3:]))
        # cmd = transformations.pose_euler_to_quaternion(cube_pose, [0, 0, 0, 0, 0, 0])
        # self.ur5_arm.set_target_pose(cmd, t=self.agent_control_dt)
        # self.rate.sleep()
        # gripper_cmd = self.scale_gripper(1.)
        # self.ur5_gripper.gripper_joint_control(gripper_cmd)


        self.ee_position_publisher.publish(str(cpose))
        self.rate.sleep()

    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        f = goal_distance(self.goal_a, self.goal_b)
        self._log_message = "Final distance error: " + str(np.round(d, 3)) \
                            + (' success!' if d < self.distance_threshold else '') \
                            + (' has object!' if self.gripper_attached else '')
        return bool(d < self.distance_threshold and f < self.distance_threshold and self.gripper_attached)