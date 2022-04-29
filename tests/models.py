from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
import numpy as np
import gym_ignition
from gym_ignition.rbd import conversions
from scipy.spatial.transform import Rotation as R
def insert_cube_in_operating_area(
        world: scenario_gazebo.World,
) -> scenario_gazebo.Model:

    # Insert objects from Fuel
    uri = lambda org, name: f"https://fuel.ignitionrobotics.org/{org}/models/{name}"

    # Download the cube SDF file
    cube_sdf = scenario_gazebo.get_model_file_from_fuel(
        uri=uri(org="openrobotics", name="wood cube 7.5cm"), use_cache=False
    )


    # Sample a random position
    #random_position = np.random.uniform(low=[0.3, -0.3, 1.01], high=[0.4, 0.3, 1.01])
    # for now the position is set to these coordinates
    random_position = np.array([0.34143738, -0.10143743, 1.01])
    #print(random_position)
    # Get a unique name
    model_name = gym_ignition.utils.scenario.get_unique_model_name(
        world=world, model_name="cube"
    )

    # Insert the model
    assert world.insert_model(
        cube_sdf, scenario_core.Pose(random_position, [1.0, 0, 0, 0]), model_name
    )

    # Return the model
    return world.get_model(model_name=model_name)


def insert_bucket(world: scenario_gazebo.World) -> scenario_gazebo.Model:

    # Insert objects from Fuel
    uri = lambda org, name: f"https://fuel.ignitionrobotics.org/{org}/models/{name}"

    # Download the cube SDF file
    bucket_sdf = scenario_gazebo.get_model_file_from_fuel(
        uri=uri(
            org="GoogleResearch",
            name="Threshold_Basket_Natural_Finish_Fabric_Liner_Small",
        ),
        use_cache=False,
    )

    # Assign a custom name to the model
    model_name = "bucket"

    # Insert the model
    assert world.insert_model(
        bucket_sdf, scenario_core.Pose([0.68, 0, 1.02], [1.0, 0, 0, 1]), model_name
    )

    # Return the model
    return world.get_model(model_name=model_name)


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