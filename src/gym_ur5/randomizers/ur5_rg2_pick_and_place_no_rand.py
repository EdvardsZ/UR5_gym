from typing import Union

from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ur5 import tasks
from gym_ur5.models import table
from gym_ur5.models.robots import ur5_rg2
from gym_ur5.models import redpoint, cube
from functools import partial
import numpy as np
import enum
# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[
    tasks.reach.Reach,
    tasks.pick_and_place.PickAndPlace,
    tasks.reach_dict.Reach
]

import time

class PickAndPLaceEnvNoRandomizations(gazebo_env_randomizer.GazeboEnvRandomizer):
    """
    Dummy environment randomizer for cartpole tasks.
    Check :py:class:`~gym_ignition_environments.randomizers.cartpole.CartpoleRandomizersMixin`
    for an example that randomizes the task, the physics, and the model.
    """

    def __init__(self, env: MakeEnvCallable):

        super().__init__(env=env)

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:
        """
        Prepare the scene for cartpole tasks. It simply removes the cartpole of the
        previous rollout and inserts a new one in the default state. Then, the active
        Task will reset the state of the cartpole depending on the implemented
        decision-making logic.
        """

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        gazebo.run(paused=True)
        # Remove the model from the simulation
        #if task.model_name is not None and task.model_name in task.world.model_names():

            #if not task.world.to_gazebo().remove_model(task.model_name):
            # #   raise RuntimeError("Failed to remove the ur5-rg2 from the world")
        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Insert a new cartpole model
        if not 'ur5_rg2' in task.world.model_names():
            model = ur5_rg2.UR5RG2(world=task.world, position=[0.5, 0.5, 1.02])
            gazebo.run(paused=True)
            model.add_ur5_controller(gazebo.step_size())
            gazebo.run(paused=True)
            task.ik = model.get_ur5_ik()
            task.model_name = model.name()
            # model.to_gazebo().enable_self_collisions(enable=True)
            #model.to_gazebo().get_link("rg2_leftfinger").to_gazebo().enable_contact_detection(True)
            #model.to_gazebo().get_link("rg2_rightfinger").to_gazebo().enable_contact_detection(True)
            gazebo.run(paused=True)
            #model.open_fingers = partial(move_fingers, _ur5_with_rg2=model, action=FingersAction.OPEN)
            #model.close_fingers = partial(move_fingers, _ur5_with_rg2=model, action=FingersAction.CLOSE)
        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")
        if not 'table' in task.world.model_names():
            table.insert(world=task.world)
        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        if 'RedPoint' in task.world.model_names():
            task.world.to_gazebo().remove_model('RedPoint')
            gazebo.run(paused=True)
            gazebo.run(paused=True)

        if 'cube' in task.world.model_names():
            task.world.to_gazebo().remove_model('cube')
            gazebo.run(paused=True)
            gazebo.run(paused=True)
        if not 'cube' in task.world.model_names():
            random_position = task.get_workspace_random_position()
            xy = random_position[:2]
            ee_pos = task.get_ee_position()[:2]
            #print("get_ee_position", ee_pos)
            while np.linalg.norm(xy - ee_pos) < 0.15:
                xy = task.get_workspace_random_position()[:2]
                #print("getting_random", xy, ee_pos)
            #random_position = [0.52555575, 0.20615784, 1.02]
            random_position[2] = 1.0575
            cube.insert(self.world, random_position)
            gazebo.run(paused=True)
            for _ in range(10): gazebo.run(paused=True)
        # if not 'cube' in task.world.model_names():
        #    position = task.workspace_centre - task.workspace_volume/2
        #    cube.insert(task.world, position)
        #    gazebo.run(paused=True)
        #    position = task.workspace_centre + task.workspace_volume/2
        #    cube.insert(task.world, position)
        #    gazebo.run(paused=True)

        if not 'RedPoint' in task.world.model_names():
            #random_position = task.get_workspace_random_position()
            random_position = [0.30143738, -0.05, 1.015]
            #print(random_position)
            redpoint.insert(self.world,random_position)


        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")
        #task.ik = model.get_ur5_ik()
        # Store the model name in the task
        #task.model_name = model.name()
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")


