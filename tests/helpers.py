#TODO bring back fingers moving fingers function
import numpy as np
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scipy.spatial.transform import Rotation as R
from gym_ignition.rbd import conversions

def solve_ik(
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        ik: inverse_kinematics_nlp.InverseKinematicsNLP,
) -> np.ndarray:

    quat_xyzw = R.from_euler(seq="xyz", angles=[0,180,0], degrees=True).as_quat()

    ik.update_transform_target(
        target_name=ik.get_active_target_names()[0],
        position=target_position,
        quaternion=conversions.Quaternion.to_wxyz(xyzw=quat_xyzw),
    )

    # Run the IK
    ik.solve()

    return ik.get_reduced_solution().joint_configuration