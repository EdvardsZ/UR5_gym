import abc
import gym
import numpy as np
from gym_ignition.base import task
from typing import Tuple
from gym_ignition.runtimes import gazebo_runtime
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

        self.workspace_centre = np.array([0.50143738, 0.15, 1.36])
        self.workspace_volume = np.array([0.4, 0.4, 0.4])

        self._is_done = False
        return

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        action_space = gym.spaces.Box(low=-1.0,
                              high=1.0,
                              shape=(3,),
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
                    -5, 5, shape=(9,), dtype="float32"
                ),
            )
        )

        return action_space, observation_space
    def set_action(self, action: Action) -> None:
        model = self.world.get_model(self.model_name)

        ee_position = self.get_ee_position()

        ee_position = ee_position + (np.array(action) * 0.1)

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

        return

    def get_observation(self) -> Observation:
        # Create the observation
        velocity = self.get_ee_velocity()
        target_pos = np.array(self.get_target_position())
        observation = np.concatenate([self.get_ee_position(), velocity, target_pos])
        observation = {
            "observation": observation.copy(),
            "achieved_goal": self.get_ee_position(),
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

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = 0.0
        distance = self.get_distance(achieved_goal, desired_goal)
        if distance < 0.05:
            reward = 1.0
        else:
            reward = -1.0
        return Reward(reward)

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
        quat_xyzw = R.from_euler(seq="xyz", angles=[0, 180, 0], degrees=True).as_quat()

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
        ee_position = self.get_ee_position()
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
        return self.world.get_model('RedPoint').base_position()

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