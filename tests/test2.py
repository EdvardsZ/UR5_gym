
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

# Process model insertion in the simulation
gazebo.run(paused=True)

# Populate the world
gazebo.run(paused=True)

while True:
    gazebo.run(paused=False)