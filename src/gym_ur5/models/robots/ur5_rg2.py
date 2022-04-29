
from typing import List, Tuple

from gym_ignition.scenario import model_with_file, model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name

from scenario import core as scenario

from scenario import gazebo as scenario_gazebo
from gym_ignition.context.gazebo import controllers

class UR5RG2(model_wrapper.ModelWrapper, model_with_file.ModelWithFile):
    def __init__(
            self,
            world: scenario.World,
            name: str = 'ur5_rg2',
            position: List[float] = (0.0, 0.0, 0.0),
            orientation: List[float] = (1.0, 0, 0, 0),
            model_file: str = None,
            use_fuel: bool = False,
            separate_gripper_controller: bool = True,
            # initial_joint_positions: List[float] = (0.0, 0.0, 1.57, 0.00000, -1.57, -1.57, 0.0, 0.0)
    ):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = UR5RG2.get_model_file(name)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file, initial_pose, model_name)

        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

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

        # Set the default PID update period
        assert model.set_controller_period(1000.0)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, robot_name) -> str:
        return 'ur5_rg2/ur5_rg2.urdf'


    # This is necessary while we dont use move it controller
    def add_ur5_controller(self,
            controller_period: float
    ) -> None:

        # Set the controller period
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

        # Insert the ComputedTorqueFixedBase controller
        assert self.to_gazebo().insert_model_plugin(
            *controllers.ComputedTorqueFixedBase(
                kp=[100.0] * (self.dofs() - 2) + [10000.0] * 2,
                ki=[0.0] * self.dofs(),
                kd=[17.5] * (self.dofs() - 2) + [100.0] * 2,
                urdf=self.get_model_file("ur5_rg2"),
                joints=list(self.joint_names()),
            ).args()
        )

        # Initialize the controller to the current state
        assert self.set_joint_position_targets(self.joint_positions())
        assert self.set_joint_velocity_targets(self.joint_velocities())
        assert self.set_joint_acceleration_targets(self.joint_accelerations())