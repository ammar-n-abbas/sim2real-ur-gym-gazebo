# Dataset

## Variables Description 

### Simulation

- `episode_steps`: Number of steps taken in an episode.
- `total_steps`: Total number of steps taken.
- `collision`: Records indicating whether a "collision" with the cell occurred during an event or if the "force" at the end effector exceeded the value of 150N.
- `object collision`: Records indicating whether a collision with an object occurred conditioned on if the object is not grasped and the force exceeds the value of 150N during an event. or collision with the obstacle.
- `speed violation`: Records indicating whether there was a violation of speed limits (max_joint_speed = np.deg2rad([170, 170, 170, 170, 170, 170])) during an event.
- `ik violation`: Records indicating whether there was a violation of inverse kinematics during an event.
- `velocity violation`: Records indicating whether there was a violation of velocity of greater than 0.2 during an event.
- `velocity during collision`: Data on velocity at the time of collision.
- `force during collision`: Data on force at the time of collision (force_x for collision with obstacle and force_z for collision with workspace or object being picked).
- `collision timestep`: Time step at which a collision occurred (if collision is True then it refers to the collision with the cell or if the force has exceeded 150N otherwise it indicates the collision with the cube).
- `velocity`: Velocity of the end effector at each step.
- `force`: Force at the end effector at each step.
- `success`: Binary indicator of overall success (grasping of object) of an event.
- `return`: Total reward per episode (epoch).


### Real

- `episode_steps`: Number of steps taken in an episode.
- `total_steps`: Total number of steps taken.
- `collision`: Records indicating whether a "collision" with the cell or object being picked occurred during an event or if the "force" at the end effector exceeded the value of 150N.
- `object collision`: Collision with the obstacle
- `speed violation`: Records indicating whether there was a violation of speed limits (max_joint_speed = np.deg2rad([170, 170, 170, 170, 170, 170])) during an event.
- `ik violation`: Records indicating whether there was a violation of inverse kinematics during an event.
- `velocity violation`: Records indicating whether there was a violation of velocity of greater than 0.2 during an event.
- `velocity during collision`: Data on velocity at the time of collision.
- `force during collision`: Data on force at the time of collision (force_x for collision with obstacle and force_z for collision with workspace or object being picked).
- `collision timestep`: Time step at which a collision occurred (if collision is True then it refers to the collision with the cell or if the force has exceeded 150N otherwise it indicates the collision with the cube).
- `velocity`: Velocity of the end effector at each step.
- `force`: Force at the end effector at each step.
- `success`: Binary indicator of overall success (grasping of object) of an event.
- `return`: Total reward per episode (epoch).
