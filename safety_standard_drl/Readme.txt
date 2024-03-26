1. collision: Records indicating whether a collision with the cell and the robot occurred during an event or if the force at the end effector exceeded the value of 150N.
2. speed violation: Records indicating whether there was a violation of speed limits (max_joint_speed = np.deg2rad([170, 170, 170, 170, 170, 170])) during an event.
3. ik violation: Records indicating whether there was a violation of inverse kinematics during an event.
4. object collision: Records indicating whether a collision with an object occurred conditioned on if the object is not grasped and the force exceeds the value of 150N during an event.
5. velocity violation: Records indicating whether there was a violation of velocity of greater than 0.2 during an event.
6. velocity during collision: Data on velocity at the time of collision.
7. success: Binary indicator of overall success (grasping of object) of an event.
8. return: total reward per episode (epoch).
