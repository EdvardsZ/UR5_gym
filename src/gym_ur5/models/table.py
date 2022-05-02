from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from gym_ignition.rbd import conversions
from scipy.spatial.transform import Rotation as R

def insert_table(world: scenario_gazebo.World) -> scenario_gazebo.Model:

    # Insert objects from Fuel
    uri = lambda org, name: f"https://fuel.ignitionrobotics.org/{org}/models/{name}"

    # Download the cube SDF file
    bucket_sdf = scenario_gazebo.get_model_file_from_fuel(
        uri=uri(org="OpenRobotics", name="Table"), use_cache=False
    )

    # Assign a custom name to the model
    model_name = "table"
    quat_xyzw = R.from_euler(seq="xyz", angles=[0, 0, -90], degrees=True).as_quat()
    # Insert the model
    assert world.insert_model(bucket_sdf, scenario_core.Pose([0.5,0.2,0],conversions.Quaternion.to_wxyz(xyzw=quat_xyzw),), model_name)

    # Return the model
    return world.get_model(model_name=model_name)