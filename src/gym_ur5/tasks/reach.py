import abc
import gym
import numpy as np
from gym_ignition.base import task
from typing import Tuple
from gym_ignition.utils.typing import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
    Reward,
)
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scipy.spatial.transform import Rotation as R
from gym_ignition.rbd import conversions
from gym_ur5.models.robots import ur5_rg2
import random
import time
from scenario import core as scenario_core
class Reach(task.Task, abc.ABC):
    def __init__(
            self, agent_rate: float, reward_cart_at_center: bool = True, **kwargs
    ) -> None:
        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)
        # Name of the cartpole model
        self.model_name = None
        self.ee_position = np.array([0.32143738, -0.10143743, 1.61])
        # Space for resetting the task
        self.reset_space = None
        self.workspace_centre = np.array([0.32143738, -0.10143743, 1.36])
        self.workspace_volume = np.array([0.4, 0.4, 0.6])

        self._is_done = False
        return

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:
        # Configure action space: [0, 1]
        action_space = gym.spaces.Box(low=-1.0,
                              high=1.0,
                              shape=(3,),
                              dtype=np.float32)


        # Configure reset limits
        high = self.workspace_centre + self.workspace_volume

        low = self.workspace_centre - self.workspace_volume

        # Configure the reset space
        self.reset_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        observation_space = gym.spaces.Box(low=low * 1.2, high=high * 1.2, dtype=np.float32)

        return action_space, observation_space
    def set_action(self, action: Action) -> None:
        model = self.world.get_model(self.model_name)
        end_effector_frame = model.get_link(link_name="tool0")

        if self.ee_position is None:
            self.ee_position = np.array([0.32143738, -0.10143743, 1.61])

        self.ee_position = self.ee_position + (np.array(action) * 0.1)

        target_pos = self.ee_position


        workspace_centre = np.array([ 0.32143738, -0.10143743, 1.36])
        workspace_volume = np.array([0.4,0.4,0.6])

        for i in range(3):
            target_pos[i] = min(workspace_centre[i] + workspace_volume[i]/2,
                             max(workspace_centre[i] - workspace_volume[i]/2,
                                 target_pos[i])
                             )

        self.ee_position = target_pos

        over_joint_configuration = self.solve_ik(
            target_position=self.ee_position,
            target_orientation=np.array([0, 1.0, 0, 0]),
            ik=self.ik,
        )
        joints = ["shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "rg2_finger_joint1",
                "rg2_finger_joint2"]
        assert model.set_joint_position_targets(over_joint_configuration, joints)

        return

    def get_observation(self) -> Observation:
        # Create the observation
        observation = Observation(self.ee_position)

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        # Calculate the reward
        reward = 1.0

        return reward

    def is_done(self) -> bool:
        done = self._is_done
        return done

    def reset_task(self) -> None:
        self._is_done = False
        self.ee_position = np.array([0.32143738, -0.10143743, 1.61])

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
            target_orientation: np.ndarray,
            ik: inverse_kinematics_nlp.InverseKinematicsNLP,
    ) -> np.ndarray:
        quat_xyzw = R.from_euler(seq="xyz", angles=[0, 180, 0], degrees=True).as_quat()

        ik.update_transform_target(
            target_name=ik.get_active_target_names()[0],
            position=target_position,
            quaternion=conversions.Quaternion.to_wxyz(xyzw=quat_xyzw),
        )

        # Run the IK
        ik.solve()

        return ik.get_reduced_solution().joint_configuration

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