# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from typing import Union

from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ur5 import tasks
from gym_ur5.models import table
#from gym_ignition_environments.models import cartpole
from gym_ur5.models.robots import ur5_rg2
from gym_ur5.models import redpoint
# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[
    tasks.reach.Reach
]


class ReachEnvNoRandomizations(gazebo_env_randomizer.GazeboEnvRandomizer):
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
                #raise RuntimeError("Failed to remove the ur5-rg2 from the world")
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

        #model.to_gazebo().enable_self_collisions(enable=True)
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
        if not 'RedPoint' in task.world.model_names():
            random_postion = task.get_workspace_random_position()
            redpoint.insert(self.world,random_postion)

        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")
        #task.ik = model.get_ur5_ik()
        # Store the model name in the task
        #task.model_name = model.name()
        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")