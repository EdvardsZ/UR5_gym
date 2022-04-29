
import time
from functools import partial
from typing import List
import random
import gym_ignition
import gym_ignition_environments
import numpy as np
from gym_ignition.context.gazebo import controllers
from gym_ignition.rbd import conversions
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scipy.spatial.transform import Rotation as R
from gym_ignition.base import task
from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
import models
from gym_ur5.models.robots import ur5_rg2
import helpers

def get_ur5_ik(
        _ur5_with_rg2: ur5_rg2.UR5RG2, optimized_joints: List[str]
) -> inverse_kinematics_nlp.InverseKinematicsNLP:

    # Create IK
    ik = inverse_kinematics_nlp.InverseKinematicsNLP(
        urdf_filename=_ur5_with_rg2.get_model_file('ur5_rg2'),
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

    return ik


def end_effector_reached(
        position: np.array,
        end_effector_link: scenario_core.Link,
        max_error_pos: float = 0.01,
        max_error_vel: float = 0.5,
        mask: np.ndarray = np.array([1.0, 1.0, 1.0]),
) -> bool:

    masked_target = mask * position
    masked_current = mask * np.array(end_effector_link.position())

    return (
            np.linalg.norm(masked_current - masked_target) < max_error_pos
            and np.linalg.norm(end_effector_link.world_linear_velocity()) < max_error_vel
    )

# ====================
# INITIALIZE THE WORLD
# ====================

# Get the simulator and the world
gazebo, world = gym_ignition.utils.scenario.init_gazebo_sim(
    step_size=0.001, real_time_factor=2.0, steps_per_run=1
)

# Open the GUI
gazebo.gui()
time.sleep(3)
gazebo.run(paused=True)

# Insert the UR5 manipulator
ur5_with_rg2 = ur5_rg2.UR5RG2(
    world=world, position=[0.5, 0.5, 1.02]
)

ur5_with_rg2.to_gazebo().enable_self_collisions(enable=True)

is_enabled_collisions = ur5_with_rg2.to_gazebo().self_collisions_enabled()
print("is_enabled: ", is_enabled_collisions)

# Enable contacts only for the finger links
ur5_with_rg2.get_link("rg2_leftfinger").to_gazebo().enable_contact_detection(True)
ur5_with_rg2.get_link("rg2_rightfinger").to_gazebo().enable_contact_detection(True)

# Process model insertion in the simulation
gazebo.run(paused=True)

# Add a custom joint controller to the UR5
ur5_with_rg2.add_ur5_controller(controller_period=gazebo.step_size())

# Populate the world
table = models.insert_table(world=world)
gazebo.run(paused=True)

# Create and configure IK for the UR5 manipulator
ik_joints = [
    j.name() for j in ur5_with_rg2.joints() if j.type is not scenario_core.JointType_fixed
]
ik = get_ur5_ik(_ur5_with_rg2=ur5_with_rg2, optimized_joints=ik_joints)

end_effector_frame = ur5_with_rg2.get_link(link_name="tool0")

initial_position = np.array([0.34143738, -0.10143743, 1.01]) + np.array([-0.02, 0, 0.6])

while True:
    gazebo.run(paused=True)

    random_position = np.array([0.34143738, -0.10143743, 1.01]) + np.array([-0.02, 0, random.uniform(0, 0.6)])

    print("Reach random position:",random_position)

    position_over_cube = random_position

    over_joint_configuration = helpers.solve_ik(
        target_position=position_over_cube,
        target_orientation=np.array([0, 1.0, 0, 0]),
        ik=ik,
    )

    # Set the joint references
    #assert ur5_with_rg2.set_joint_position_targets(over_joint_configuration, ik_joints)

    # Run the simulation until the EE reached the desired position
   # while not end_effector_reached(
   #          position=position_over_cube,
   #          end_effector_link=end_effector_frame,
   #          max_error_pos=0.5,
   #          max_error_vel=0.5,
   # ):
   #     gazebo.run()

    # Wait a bit more
    [gazebo.run() for _ in range(500)]