#!/usr/bin/env python3

############# ROS Dependencies #####################################
import rospy
import os
import threading
import tf_conversions as tf_c

from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32, Float64, Int16, String, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Joy, JointState, PointCloud, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy
from numpy import sum
from tf.transformations import quaternion_matrix
from ipaddress import collapse_addresses
from itertools import chain
import queue
from re import T
from numpy.core.numeric import cross
from geometry_msgs import msg
import rospy
from numpy import matrix, matmul, transpose, isclose, array, rad2deg, abs, vstack, hstack, shape, eye, zeros
from threading import Thread, Lock
from numpy.linalg import norm, det
from math import atan2, pi, asin, acos
from copy import copy
from numpy import flip
from tf2_ros import TransformBroadcaster

# Plot for ROS - Geometry message
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
import PyKDL
from urdf_parser_py.urdf import URDF

# from kdl_parser_py import KDL
from kdl_parser_py import urdf

#import open3d as o3d
### For service - Fixture line detection
from std_srvs.srv import Trigger, TriggerRequest
from std_msgs.msg import Float64MultiArray,Int32
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.joint_kinematics import JointKinematics

import numpy as np

# import roslib
# roslib.load_manifest('joint_states_listener')
# from joint_states_listener.srv import *
  
# For joint angle in URDF to screw
mutex = Lock()

# Import URDF of the robot - # todo the param file for fetching the URDF without file location

## Find the appropriate urdf here
robot_description = URDF.from_xml_file(
    '/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_description/urdf/ur5_imr.urdf')

robot_urdf = URDF.from_xml_file(
    '/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_description/urdf/ur5_imr.urdf')

# Build the tree here from the URDF parser file from the file location
build_ok, kdl_tree = urdf.treeFromFile(
    '/home/imr/Desktop/ur5_drl_ammar/ws_ur_openai_ros_imr/src/ur5/ur_description/urdf/ur5_imr.urdf')

if build_ok == True:
    print('KDL chain built successfully !!')
else:
    print('KDL chain unsuccessful')

base_link = "base_link"
tip_link = "tool0"

robot_suffix = ""

# Build the kdl_chain here
kdl_chain = kdl_tree.getChain(base_link, tip_link)

##############################

# PyKDL_Util here
pykdl_util_kin = KDLKinematics(robot_description, base_link, tip_link)

# Differential jacobian here
vel_ik_solver = PyKDL.ChainIkSolverVel_pinv(kdl_chain, 0.0001, 1000)


class URArm():
    def __init__(self):
        # self.lock = threading.Lock()
        # self.thread = threading.Thread(target=self.joint_states_listener)
        # self.thread.start()
        self.desired_pose = Pose()
        self.no_of_joints = kdl_chain.getNrOfJoints()
        self.q_in = PyKDL.JntArray(kdl_chain.getNrOfJoints())
        self.q_in_numpy = zeros(6)

        rospy.Subscriber("/robot/joint_states", JointState, self.joint_states_callback)
        self.joint_angles = [0, 0, 0, 0, 0, 0]

        self.ur_joint_trajectory_publisher = rospy.Publisher("/joint_trajectory_controller/command", JointTrajectory, queue_size=1)
        # self.fanuc_joint1_position_publisher = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        # self.fanuc_joint2_position_publisher = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
        # self.fanuc_joint3_position_publisher = rospy.Publisher("/joint3_position_controller/command", Float64, queue_size=1)
        # self.fanuc_joint4_position_publisher = rospy.Publisher("/joint4_position_controller/command", Float64, queue_size=1)
        # self.fanuc_joint5_position_publisher = rospy.Publisher("/joint5_position_controller/command", Float64, queue_size=1)
        # self.fanuc_joint6_position_publisher = rospy.Publisher("/joint6_position_controller/command", Float64, queue_size=1)
        
        self.end_effector_pos = rospy.Publisher("/end_effector_pos",PoseStamped,queue_size=1)
        self.end_effector_position = rospy.Publisher("/end_effector_position",PointStamped,queue_size=1)

        self.scalerXYZ = [0.5, 0.5, 0.5]
       
        self.pub_rate = 1000  # Hz
        self.vel_scale = array([5.0, 5.0, 5.0])

        self.cartesian_twist = PyKDL.Twist()
        self.twist_output = PyKDL.Twist()
        self.test_output_twist = PyKDL.Twist()
        self.qdot_out = PyKDL.JntArray(kdl_chain.getNrOfJoints())

        # SVD damped
        self.vel_ik_solver = PyKDL.ChainIkSolverVel_wdls(kdl_chain, 0.00001, 150)

        self.vel_ik_pinv_solver = PyKDL.ChainIkSolverVel_pinv(kdl_chain, 0.00001, 150)

        self.vel_fk_solver = PyKDL.ChainFkSolverPos_recursive(kdl_chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(kdl_chain)

        self.eeFrame = PyKDL.Frame()
        self.end_effector_pos = zeros(shape=(3))
        self.fk_jacobian = PyKDL.Jacobian(self.no_of_joints)
        self.ik_jacobian_K = PyKDL.Jacobian(self.no_of_joints)

        # Numpy jacobian array
        self.fk_jacobian_arr = zeros(shape=(self.no_of_joints, self.no_of_joints))

        br1 = TransformBroadcaster()

        # Limits of all jointts are here
        self.robot_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.robot_joint_names_pub = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        self.q_upper_limit = [robot_description.joint_map[i].limit.upper - 0.07 for i in self.robot_joint_names]
        self.q_lower_limit = [robot_description.joint_map[i].limit.lower + 0.07 for i in self.robot_joint_names]

        self.qdot_limit = [robot_description.joint_map[i].limit.velocity for i in self.robot_joint_names]

        # Gripper state message declaration
        self.gripper_state_msg = Bool()
 
        self.eeFrame = kdl_chain.getSegment(0).getFrameToTip()
        self.baseFrame = PyKDL.Frame.Identity()

        self.qdot_limit = [robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]

        self.qdot_max = array(self.qdot_limit)
        self.qdot_min = -1*self.qdot_max

        self.q_in = zeros(6)

        self.pykdl_util_kin = KDLKinematics(robot_urdf, base_link, tip_link, None)

        self.q_upper_limit = array([self.pykdl_util_kin.joint_limits_upper]).T
        self.q_lower_limit = array([self.pykdl_util_kin.joint_limits_lower]).T

        self.q_bounds = hstack((self.q_lower_limit, self.q_upper_limit))

        ############# Python Attributes ####################################    
        
    def joint_trajectory_publisher_robot(self, q_joints, duration=5):
        q_in = copy(q_joints)

        jt = JointTrajectory()
        jt.joint_names = [self.robot_joint_names_pub[0], self.robot_joint_names_pub[1],self.robot_joint_names_pub[2],self.robot_joint_names_pub[3],
                          self.robot_joint_names_pub[4], self.robot_joint_names_pub[5]]
        jt.header.stamp = rospy.Time.now()
        jt.header.frame_id = ''

        jtpt = JointTrajectoryPoint()
        jtpt.positions = [q_in[0], q_in[1], q_in[2], q_in[3], q_in[4], q_in[5]]
        jtpt.time_from_start = rospy.Duration.from_sec(duration)
        jt.points.append(jtpt)

        self.ur_joint_trajectory_publisher.publish(jt) 

    # def joint_position_publisher(self, q_joints):
    #     self.fanuc_joint1_position_publisher.publish(q_joints[0])
    #     self.fanuc_joint2_position_publisher.publish(q_joints[1])
    #     self.fanuc_joint3_position_publisher.publish(q_joints[2])
    #     self.fanuc_joint4_position_publisher.publish(q_joints[3])
    #     self.fanuc_joint5_position_publisher.publish(q_joints[4])
    #     self.fanuc_joint6_position_publisher.publish(q_joints[5])
    
    # callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        mutex.acquire()
        self.name = msg.name
        self.joint_angles = msg.position       
        self.joint_vels = msg.velocity
        mutex.release()

    def ee_position(self):
        q_in = self.joint_angles
        mutex.acquire()
        pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3, 3])
        pos_matrix = array(self.pykdl_util_kin.forward(q_in))

        pointmsg = PointStamped()

        pointmsg.point.x = pos_act[0]
        pointmsg.point.y = pos_act[1]
        pointmsg.point.z = pos_act[2]
        self.end_effector_position.publish(pointmsg)

        J_Hess = array(self.pykdl_util_kin.jacobian(q_in))
        mutex.release()
        return q_in, pos_act
    
    def j_pos_inv_kin(self, pose):
        mutex.acquire()
        q_out = array(self.pykdl_util_kin.inverse(pose))
        mutex.release()
        return q_out
