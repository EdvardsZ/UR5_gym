import abc
import numpy as np
from gym_ignition.base import task
from typing import Tuple
from gym_ignition.utils.typing import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
    Reward,
    Dict
)
import gym
from gym import error, spaces
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scipy.spatial.transform import Rotation as R
from gym_ignition.rbd import conversions
from scenario import core as scenario_core
from gym_ur5.models import cube


class PickAndPlace(task.Task, abc.ABC):
    def __init__(
            self, agent_rate: float, reward_cart_at_center: bool = True, **kwargs
    ) -> None:
        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)
        # Name of the cartpole model
        self.model_name = None
        self.finger_state = None # 0 for open 1 closed
        self.workspace_centre = np.array([0.50143738, 0.05, 1.27])
        self.workspace_volume = np.array([0.4, 0.4, 0.5])

        self._is_done = False
        return

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        action_space = gym.spaces.Box(low=-1.0,
                              high=1.0,
                              shape=(4,),
                              dtype=np.float32)
        observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -5, 5, shape=(3,), dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -5, 5, shape=(3,), dtype="float32"
                ),
                observation=spaces.Box(
                    -5, 5, shape=(31,), dtype="float32"
                ),
            )
        )

        return action_space, observation_space

    def set_action(self, action: Action) -> None:
        model = self.world.get_model(self.model_name)
        ee_position = self.get_ee_position()
        ee_position = ee_position + (np.array(action[:3]) * 0.15)

        target_pos = ee_position

        for i in range(3):
            target_pos[i] = min(self.workspace_centre[i] + self.workspace_volume[i]/2,
                            max(self.workspace_centre[i] - self.workspace_volume[i]/2,
                                target_pos[i])
                            )

        ee_position = target_pos

        over_joint_configuration = self.solve_ik(
            target_position=ee_position,
            ik=self.ik,
        )
        joints = self.get_joints()
        assert model.set_joint_position_targets(over_joint_configuration, joints)
        self.move_fingers(action[3])

        return

    def get_observation(self) -> Observation:
        # Create the observation
        velocity = self.get_ee_velocity()
        target_pos = self.get_target_position()
        cube_pos = self.get_cube_position()
        cube_relative_to_gripper = cube_pos - self.get_ee_position()
        gripper_angles = self.get_fingers_angles()
        gripper_velocity = self.get_fingers_angular_velocity()
        cube_angle = self.get_cube_angle()
        cube_angular_velocity = self.get_cube_angular_velocity()
        cube_velocity = self.get_cube_velocity()
        contact = self.get_contact_wrench()
        observation = np.concatenate([self.get_ee_position(), cube_pos, cube_relative_to_gripper, gripper_angles, gripper_velocity, cube_angle, cube_angular_velocity, cube_velocity, contact, velocity, target_pos])

        observation = {
            "observation": observation.copy(),
            "achieved_goal": cube_pos,
            "desired_goal": target_pos,
        }
        # Return the observation
        return Observation(observation)

    def get_reward(self) -> Reward:
        reward = 0.0
        distance = self.get_distance_to_target()
        if distance < 0.05:
            reward = 1.0
        else:
            reward = -1.0
        return Reward(reward)

    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = self.goal_distance(achieved_goal, desired_goal)
        result = -(d > 0.05).astype(np.float32)
        result = np.where( result == 0.0, 1.0, -1.0)
        return result

    def get_distance(self, position, goal):
        # Get current end-effector and target positions
        ee_position = position
        target_position = goal

        # Compute the current distance to the target
        return np.linalg.norm([ee_position[0] - target_position[0],
                               ee_position[1] - target_position[1],
                               ee_position[2] - target_position[2]])

    def get_info(self) -> Dict:
        distance = self.get_distance_to_target()
        success = False if distance > 0.05 else True
        info = {
            "is_success": success,
        }
        return info

    def is_done(self) -> bool:
        done = self._is_done
        return done

    def reset_task(self) -> None:
        self._is_done = False

        model = self.world.get_model(self.model_name)
        joint_config = [1.47838380e+00, -2.15699582e+00, -8.14691050e-05, -2.52916159e+00, 1.56735617e+00, -9.25439234e-02, 0, 0]
        # Set the joint references
        assert model.set_joint_position_targets(joint_config, self.get_joints())

        return

    def get_joints(self):
        return ["shoulder_pan_joint",
         "shoulder_lift_joint",
         "elbow_joint",
         "wrist_1_joint",
         "wrist_2_joint",
         "wrist_3_joint",
         "rg2_finger_joint1",
         "rg2_finger_joint2"]

    def solve_ik(
            self,
            target_position: np.ndarray,
            ik: inverse_kinematics_nlp.InverseKinematicsNLP,
    ) -> np.ndarray:
        quat_xyzw = R.from_euler(seq="xyz", angles=[0, 180, 90], degrees=True).as_quat()

        ik.update_transform_target(
            target_name=ik.get_active_target_names()[0],
            position=target_position,
            quaternion=conversions.Quaternion.to_wxyz(xyzw=quat_xyzw),
        )

        # Run the IK
        ik.solve()

        return ik.get_reduced_solution().joint_configuration

    def get_distance_to_target(self):
        # Get current end-effector and target positions
        ee_position = self.get_cube_position() # changed this to cube position for this environmnet
        target_position = np.array(self.get_target_position())

        # Compute the current distance to the target
        return np.linalg.norm([ee_position[0] - target_position[0],
                               ee_position[1] - target_position[1],
                               ee_position[2] - target_position[2]])

    def end_effector_reached(self,
            position: np.array,
            end_effector_link: scenario_core.Link,
            max_error_pos: float = 0.25,
            max_error_vel: float = 0.5,
            mask: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ) -> bool:
        masked_target = mask * position
        masked_current = mask * np.array(end_effector_link.position())

        return (
                np.linalg.norm(masked_current - masked_target) < max_error_pos
                and np.linalg.norm(end_effector_link.world_linear_velocity()) < max_error_vel
        )
    def get_target_position(self):
        return np.array(self.world.get_model('RedPoint').base_position())

    def get_cube_position(self):
        position = np.array(self.world.get_model('cube').base_position())
        if(position[2] < 1.0):
            self.respawn_cube(position)
        return position

    def get_workspace_random_position(self):
        low = self.workspace_centre - self.workspace_volume/2
        high = self.workspace_centre + self.workspace_volume/2
        point = np.random.uniform(low, high, size=3)
        return point

    def get_ee_position(self):
        model = self.world.get_model(self.model_name).to_gazebo()
        return np.array(model.get_link('tool0').position())
    def get_ee_velocity(self):
        model = self.world.get_model(self.model_name).to_gazebo()
        return np.array(model.get_link('tool0').world_linear_velocity())

    def get_fingers_angles(self):
        model = self.world.get_model(self.model_name).to_gazebo()
        finger1 = model.get_joint(joint_name="rg2_finger_joint1")
        finger2 = model.get_joint(joint_name="rg2_finger_joint2")
        return np.array([finger1.position(), finger2.position()])
    def get_fingers_angular_velocity(self):
        model = self.world.get_model(self.model_name).to_gazebo()
        finger1 = model.get_joint(joint_name="rg2_finger_joint1")
        finger2 = model.get_joint(joint_name="rg2_finger_joint2")
        return np.array([finger1.velocity(), finger2.velocity()])

    def get_cube_angle(self):
        model = self.world.get_model('cube').to_gazebo()
        return np.array(model.base_orientation())

    def get_cube_angular_velocity(self):
        model = self.world.get_model('cube').to_gazebo()
        return np.array(model.base_body_angular_velocity())

    def get_cube_velocity(self):
        model = self.world.get_model('cube').to_gazebo()
        return np.array(model.base_body_linear_velocity())

    def get_contact_wrench(self):
        model = self.world.get_model(self.model_name).to_gazebo()
        finger_left = model.get_link(link_name="rg2_leftfinger")
        finger_right = model.get_link(link_name="rg2_rightfinger")
        return np.array([np.linalg.norm(finger_left.contact_wrench()), np.linalg.norm(finger_right.contact_wrench())])

    def move_fingers(self, action=0.0
    ) -> None:
        action = (action + 1.0) / 2
        model = self.world.get_model(self.model_name).to_gazebo()
        # Get the joints of the fingers
        finger1 = model.get_joint(joint_name="rg2_finger_joint1")
        finger2 = model.get_joint(joint_name="rg2_finger_joint2")
        # because open = 1.18 and closed = 0.0
        position = action * 1.18
        finger1.set_position_target(position=position)
        finger2.set_position_target(position=position)

    def respawn_cube(self, position):
        if 'cube' in self.world.model_names():
            position[2] = 1.05
            self.world.to_gazebo().remove_model('cube')
            self.gazebo.run()
            grasping_object = cube.insert(self.world, position)
            grasping_object = grasping_object.to_gazebo()
            grasping_object.set_base_world_linear_velocity_target([0.0, 0.0, 0.0])
            for _ in range(20): self.gazebo.run(paused=True)
            self.gazebo.run()
        return



