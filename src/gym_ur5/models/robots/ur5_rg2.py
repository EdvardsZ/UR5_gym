from gym_ignition.scenario import model_wrapper, model_with_file
from gym_ignition.utils.scenario import get_unique_model_name
from gym_ignition.context.gazebo import controllers
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from typing import List, Tuple
from os import path
import numpy as np
from scenario import core as scenario_core


class UR5RG2(model_wrapper.ModelWrapper,
             model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'ur5_rg2',
                 position: List[float] = (0, 0, 1),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_file: str = None,
                 use_fuel: bool = True,
                 arm_collision: bool = True,
                 hand_collision: bool = True,
                 separate_gripper_controller: bool = True,
                 initial_joint_positions: List[float] = (0.0, 0.0, 1.57, 0.0, -1.57, -1.57, 0.0, 0.0)):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(False)

       # if not arm_collision or not hand_collision:
       #     model_file = self.disable_collision(model_file=model_file,
       #                                         arm_collision=arm_collision,
        #                                        hand_collision=hand_collision)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        #self.__separate_gripper_controller = separate_gripper_controller
        if not model.to_gazebo().reset_joint_positions(
            [0., -1.571, 0., -1.571, 0, 1.571],
            [name for name in model.joint_names() if name not in ["rg2_finger_joint1", "rg2_finger_joint2"]]
        ):
            raise RuntimeError("Failed to set initial robot joint positions")

        pid_gains_1000hz = {
            "shoulder_pan_joint": scenario.PID(50, 0, 20),
            "shoulder_lift_joint": scenario.PID(10000, 0, 500),
            "elbow_joint": scenario.PID(100, 0, 10),
            "wrist_1_joint": scenario.PID(1000, 0, 50),
            "wrist_2_joint": scenario.PID(100, 0, 10),
            "wrist_3_joint": scenario.PID(100, 0, 10),
            "rg2_finger_joint1": scenario.PID(100, 0, 50),
            "rg2_finger_joint2": scenario.PID(100, 0, 50),
        }

        # Check that all joints have gains
        if not set(model.joint_names()) == set(pid_gains_1000hz.keys()):
            raise ValueError("The number of PIDs does not match the number of joints")

        # Set the PID gains
        for joint_name, pid in pid_gains_1000hz.items():

            if not model.get_joint(joint_name).set_pid(pid=pid):
                raise RuntimeError(f"Failed to set the PID of joint '{joint_name}'")

        assert model.set_controller_period(1000.0)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=False) -> str:
        if fuel:
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/ur5_rg2")
        else:
            return "ur5_rg2/ur5_rg2.urdf"

    @classmethod
    def get_joint_names(self) -> List[str]:
        return ["shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "rg2_finger_joint1",
                "rg2_finger_joint2"]

    @classmethod
    def get_joint_limits(self) -> List[Tuple[float, float]]:
        return [(-6.28319, 6.28319),
                (-6.28319, 6.28319),
                (-6.28319, 6.28319),
                (-6.28319, 6.28319),
                (-6.28319, 6.28319),
                (-6.28319, 6.28319),
                (0.0, 0.52359878),
                (0.0, 0.52359878)]

    @classmethod
    def get_base_link_name(self) -> str:
        return "base_link"

    @classmethod
    def get_ee_link_name(self) -> str:
        return "tool0"

    @classmethod
    def get_gripper_link_names(self) -> List[str]:
        return ["rg2_leftfinger",
                "rg2_rightfinger"]

    @classmethod
    def get_finger_count(self) -> int:
        return 2

    def get_initial_joint_positions(self) -> List[float]:
        return self.__initial_joint_positions

    def __set_initial_joint_positions(self, initial_joint_positions):
        self.__initial_joint_positions = initial_joint_positions

    def add_ur5_controller(self, controller_period: float
    ) -> None:

        # Set the controller period # TODO set the period
        assert self.set_controller_period(period=controller_period)

        self.get_joint(
            joint_name="shoulder_pan_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="shoulder_lift_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="elbow_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="wrist_1_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="wrist_2_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="wrist_3_joint"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)

        # Increase the max effort of the fingers
        self.get_joint(
            joint_name="rg2_finger_joint1"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)
        self.get_joint(
            joint_name="rg2_finger_joint2"
        ).to_gazebo().set_max_generalized_force(max_force=500.0)

        # Insert the ComputedTorqueFixedBase controller
        assert self.to_gazebo().insert_model_plugin(
            *controllers.ComputedTorqueFixedBase(
                kp=[100.0] * (self.dofs() - 2) + [10000.0] * 2,
                ki=[0.0] * self.dofs(),
                kd=[17.5] * (self.dofs() - 2) + [100.0] * 2,
                urdf=self.get_model_file(),
                joints=list(self.joint_names()),
            ).args()
        )

        # Initialize the controller to the current state
        assert self.set_joint_position_targets(self.joint_positions())
        assert self.set_joint_velocity_targets(self.joint_velocities())
        assert self.set_joint_acceleration_targets(self.joint_accelerations())

    def get_ur5_ik(
            self
    ) -> inverse_kinematics_nlp.InverseKinematicsNLP:
        optimized_joints = self.get_joint_names()
        # Create IK
        ik = inverse_kinematics_nlp.InverseKinematicsNLP(
            urdf_filename=self.get_model_file(),
            considered_joints=optimized_joints,
            joint_serialization=list(self.joint_names()),
        )

        # Initialize IK
        ik.initialize(
            verbosity=1,
            floating_base=False,
            cost_tolerance=1e-8,
            constraints_tolerance=1e-8,
            base_frame=self.base_frame(),
        )

        # Set the current configuration
        ik.set_current_robot_configuration(
            base_position=np.array(self.base_position()),
            base_quaternion=np.array(self.base_orientation()),
            joint_configuration=np.array(self.joint_positions()),
        )

        # Add the cartesian target of the end effector
        # end_effector = "rg2_hand"
        end_effector = "tool0"
        ik.add_target(
            frame_name=end_effector,
            target_type=inverse_kinematics_nlp.TargetType.POSE,
            as_constraint=False,
        )

        self.ik = ik

        return ik
