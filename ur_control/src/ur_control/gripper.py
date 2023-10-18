import rospy
from std_msgs.msg import Float64


class Gripper(object):
    """ UR5 arm controller """

    def __init__(self):
        self.gripper_joint_position_publisher = rospy.Publisher("/gripper_joint_position_controller/command", Float64, queue_size=1)

         
    def gripper_joint_control(self, gripper_cmd):
         self.gripper_joint_position_publisher.publish(gripper_cmd)
 