from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from gym_ignition.rbd import conversions
from scipy.spatial.transform import Rotation as R

def insert(world: scenario_gazebo.World, position) -> scenario_gazebo.Model:

    # Insert objects from Fuel
    uri = lambda org, name: f"https://fuel.ignitionrobotics.org/1.0/{org}/models/{name}"

    # Download the cube SDF file
    red_point_sdf = scenario_gazebo.get_model_file_from_fuel(
        uri=uri(org="EdvardsZ", name="RedPoint"), use_cache=False
    )
    # Assign a custom name to the model
    model_name = "RedPoint"
    # Insert the model
    assert world.insert_model(red_point_sdf, scenario_core.Pose(position,[1.0,0,0,0]), model_name)

    # Return the model
    return world.get_model(model_name=model_name)