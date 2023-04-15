import pybullet as p
import numpy as np
import argparse
import yaml
import os
cwd = os.getcwd()

# from robomimic.utils.file_utils import policy_from_checkpoint
# from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
# from config.atlas_config import PnCConfig
from util import pybullet_util
from config.atlas_config import SimConfig

# Environment Setup
p.connect(p.GUI)
p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                cameraYaw=120,
                                cameraPitch=-30,
                                cameraTargetPosition=[1, 0.5, 1.5])
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                            numSubSteps=SimConfig.N_SUBSTEP)
# if SimConfig.VIDEO_RECORD:
# video_dir = 'video/atlas_pnc'
# if os.path.exists(video_dir):
# shutil.rmtree(video_dir)
# os.makedirs(video_dir)

# Create Robot, Ground
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
def createRobot():
    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
                        SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)


    return(robot)

#######################################################################