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
import numpy as np
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
        self.set_position = np.array([0.32143738, -0.10143743, 1.61])
        self.random_position = self.set_position
        # Space for resetting the task
        self.reset_space = None

        # Private attributes
        self._force_mag = 20.0  # Nm
        self._reward_cart_at_center = reward_cart_at_center

        # Variables limits
        self._x_threshold = 2.4  # m
        self._dx_threshold = 20.0  # m /s
        self._q_threshold = np.deg2rad(12)  # rad
        self._dq_threshold = np.deg2rad(3 * 360)  # rad / s
        return

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:
        # Configure action space: [0, 1]
        action_space = gym.spaces.Box(low=-1.0,
                              high=1.0,
                              shape=(3,),
                              dtype=np.float32)


        # Configure reset limits
        high = np.array(
            [
                self._x_threshold,  # x
                self._dx_threshold,  # dx
                self._q_threshold,  # q
                self._dq_threshold,  # dq
            ]
        )

        # Configure the reset space
        self.reset_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        # Configure the observation space
        obs_high = high.copy() * 1.2
        observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        return action_space, observation_space
    def set_action(self, action: Action) -> None:
        #print("Action")
        model = self.world.get_model(self.model_name)
        end_effector_frame = model.get_link(link_name="tool0")
        if self.set_position is None:
            self.set_position = np.array([0.32143738, -0.10143743, 1.61])
        if self.random_position is None:
            self.random_position = np.array([0.32143738, -0.10143743, 1.61])
        #print(action)
        self.random_position = self.random_position + (np.array(action) * 0.05)

        target_pos = self.random_position

        workspace_centre = np.array([ 0.32143738, -0.10143743, 1.36])
        workspace_volume = np.array([0.4,0.4,0.6])

        for i in range(3):
            target_pos[i] = min(workspace_centre[i] + workspace_volume[i]/2,
                             max(workspace_centre[i] - workspace_volume[i]/2,
                                 target_pos[i])
                             )

        self.random_position = target_pos
        #print(self.random_position)
        if (self.end_effector_reached(self.set_position,end_effector_frame)):# move it to reach position only if position reached or too many inverse kinematics calculations
            start_time = time.time()
            self.set_position = self.random_position
            over_joint_configuration = self.solve_ik(
                target_position=self.random_position,
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
            # Set the joint references
            assert model.set_joint_position_targets(over_joint_configuration, joints)
            print("--- %s seconds ---" % (time.time() - start_time))


        # Calculate the force
        #force = self._force_mag if action == 1 else -self._force_mag

        # Set the force value
       # model = self.world.get_model(self.model_name)
        #ok_force = model.get_joint("linear").set_generalized_force_target(force)

        #if not ok_force:
            #raise RuntimeError("Failed to set the force to the cart")
        return

    def get_observation(self) -> Observation:

        # Get the model
       # model = self.world.get_model(self.model_name)

        # Get the new joint positions and velocities
      #  q, x = model.joint_positions(["pivot", "linear"])
      #  dq, dx = model.joint_velocities(["pivot", "linear"])

        # Create the observation
        observation = Observation(np.array([1, 2, 3, 4]))

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        # Calculate the reward
        reward = 1.0

        return reward

    def is_done(self) -> bool:

        # Get the observation
        #observation = self.get_observation()

        # The environment is done if the observation is outside its space
        #done = not self.reset_space.contains(observation)

        return False

    def reset_task(self) -> None:
        self.random_position = None
        self.set_position = None
        return
        # if self.model_name not in self.world.model_names():
        #     raise RuntimeError("Cartpole model not found in the world")
        #
        # # Get the model
        # model = self.world.get_model(self.model_name)
        #
        # # Control the cart in force mode
        # linear = model.get_joint("linear")
        # ok_control_mode = linear.set_control_mode(scenario.JointControlMode_force)
        #
        # if not ok_control_mode:
        #     raise RuntimeError("Failed to change the control mode of the cartpole")
        #
        # # Create a new cartpole state
        # x, dx, q, dq = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #
        # # Reset the cartpole state
        # ok_reset_pos = model.to_gazebo().reset_joint_positions(
        #     [x, q], ["linear", "pivot"]
        # )
        # ok_reset_vel = model.to_gazebo().reset_joint_velocities(
        #     [dx, dq], ["linear", "pivot"]
        # )
        #
        # if not (ok_reset_pos and ok_reset_vel):
        #     raise RuntimeError("Failed to reset the cartpole state")

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

    def get_ur5_ik(self,
            _ur5_with_rg2: ur5_rg2.UR5RG2
    ) -> inverse_kinematics_nlp.InverseKinematicsNLP:
        optimized_joints = _ur5_with_rg2.get_joint_names()
        # Create IK
        ik = inverse_kinematics_nlp.InverseKinematicsNLP(
            urdf_filename=_ur5_with_rg2.get_model_file(),
            considered_joints=optimized_joints,
            joint_serialization=list(_ur5_with_rg2.joint_names()),
        )

        # Initialize IK
        ik.initialize(
            verbosity=1,
            floating_base=False,
            cost_tolerance=1e-8,
            constraints_tolerance=1e-8,
            base_frame=_ur5_with_rg2.base_frame(),
        )

        # Set the current configuration
        ik.set_current_robot_configuration(
            base_position=np.array(_ur5_with_rg2.base_position()),
            base_quaternion=np.array(_ur5_with_rg2.base_orientation()),
            joint_configuration=np.array(_ur5_with_rg2.joint_positions()),
        )

        # Add the cartesian target of the end effector
        # end_effector = "rg2_hand"
        end_effector = "tool0"
        ik.add_target(
            frame_name=end_effector,
            target_type=inverse_kinematics_nlp.TargetType.POSE,
            as_constraint=False,
        )

        _ur5_with_rg2.ik = ik

        return ik

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