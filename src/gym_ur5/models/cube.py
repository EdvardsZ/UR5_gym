from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
import numpy as np
import gym_ignition
from gym_ignition.rbd import conversions
from scipy.spatial.transform import Rotation as R
def insert(
        world: scenario_gazebo.World, position
) -> scenario_gazebo.Model:

    # Insert objects from Fuel
    uri = lambda org, name: f"https://fuel.ignitionrobotics.org/{org}/models/{name}"

    # Download the cube SDF file
    cube_sdf = scenario_gazebo.get_model_file_from_fuel(
        uri=uri(org="openrobotics", name="wood cube 6cm"), use_cache=False
    )

    model_name = gym_ignition.utils.scenario.get_unique_model_name(
        world=world, model_name="cube"
    )

    # Insert the model
    assert world.insert_model(
        cube_sdf, scenario_core.Pose(position, [1.0, 0, 0, 0]), model_name
    )

    # Return the model
    return world.get_model(model_name=model_name)