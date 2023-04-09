import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil
import environments
import argparse
import yaml
from robomimic.utils.file_utils import policy_from_checkpoint

import pybullet as p
import numpy as np
np.set_printoptions(precision=2)

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import pybullet_util
from util import util
from util import liegroup

DEG_TO_RAD = np.pi/180.
RAD_TO_DEG = 180./np.pi

def set_initial_config(robot, joint_id):
    # shoulder_x
    p.resetJointState(robot, joint_id["l_arm_shx"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_arm_shx"], np.pi / 4, 0.)
    # elbow_y
    p.resetJointState(robot, joint_id["l_arm_ely"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_ely"], np.pi / 2, 0.)
    # elbow_x
    p.resetJointState(robot, joint_id["l_arm_elx"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_elx"], -np.pi / 2, 0.)
    # hip_y
    p.resetJointState(robot, joint_id["l_leg_hpy"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_hpy"], -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, joint_id["l_leg_kny"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_leg_kny"], np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, joint_id["l_leg_aky"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_aky"], -np.pi / 4, 0.)


def signal_handler(signal, frame):
    # if SimConfig.VIDEO_RECORD:
    # pybullet_util.make_video(video_dir)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

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
    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
                       SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    planeId = p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)
    
    #######################################################################
    
    FIELD_RANGE = [50, 3]
    FURNITURES = {}
    OBJECTS = {}
    assets = {}
    mass = 0
    setup_obstacles = {}
    p.changeVisualShape(planeId, -1, textureUniqueId=p.loadTexture("data1/textures/carpet.png"), rgbaColor=(0.8,0.7,0.8,1.0))
    setup_obstacles = environments.PerceptionWrapper._generate_obstacle_profiles(p,10,10,10,50,2)
    assets = environments.PerceptionWrapper._set_world(p,setup_obstacles)

    # chairId = p.loadURDF(cwd + "/data1/assets/furnitures/chair_1/model.urdf", [2.5, 0, 0])
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    #     robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
    #     SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    
    #####################################################################################

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    # Construct Interface
    interface = AtlasInterface()


    ############################################################################################ (create eval_policy)
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav_policy", type=str, default="bcrnn",
                    help="path for loading checkpoints, configuration and training logs. For example, --nav_policy=NAV_POLICY will load checkpoints at ./save/bc_checkpoints/NAV_POLICY.")
    parser.add_argument("--config",type=str,default="deploy",
                        help="path to a directory with the configuration files of simulation setup, etc. For example, --config=CONFIG will save checkpoints at ./config/CONFIG.")
    args = parser.parse_args()
    nav_policy = args.nav_policy
    PATH_SICRIPT    = os.path.dirname(os.path.realpath(__file__))
    PATH_ROOT   = os.path.dirname(PATH_SICRIPT)
    SUBPATH = yaml.load(open(os.path.join(PATH_ROOT, 'path.yaml')), Loader=yaml.FullLoader)
    PATH_CHECKPOINT_BC = os.path.join(PATH_ROOT, SUBPATH['BC Checkpoint'])
    nav_path = "{}/{}/models/model_best_training.pth".format(PATH_CHECKPOINT_BC, nav_policy)
    eval_policy = policy_from_checkpoint(ckpt_path=nav_path)[0]
    # print(eval_policy)

    ######################################################################################### (get obs)
    _view_agent = {'dist': 0.2,
                            'offset': np.array([0.45, 0, 0.]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': 0. * DEG_TO_RAD,
                            'width': 212,
                            'height': 120,
                            'near': 0.1,
                            'far': 100}

    _view_bird = {'dist': 2.0,
                            'offset': np.array([0.45, 0, 0]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -60. * DEG_TO_RAD,
                            'pitch': -30 * DEG_TO_RAD,
                            'width': 480,
                            'height': 360,
                            'near': 0.1,
                            'far': 100}

    _view_fpv = {'dist': 1.5,
                            'offset': np.array([0.45, 0, 0]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': -30 * DEG_TO_RAD,
                            'width': 480,
                            'height': 360,
                            'near': 0.1,
                            'far': 100}
    

    ############################################################################################

    
    

    ##############################################################################################

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0
    _nav_action = 0
    camera_distance = 3  # distance from the robot
    camera_yaw = -90  # camera's yaw angle (around the vertical axis)
    camera_pitch = 45  # camera's pitch angle (around the horizontal axis)
    camera_target_position = p.getBasePositionAndOrientation(robot)[0]  # target position is the robot's base position
    #camera_target_position[2] += 1  # offset the camera's height above the robot
    camera_orientation = p.getQuaternionFromEuler([camera_pitch, camera_yaw, 0])  # camera's orientation (in quaternion)

    while (1):

        # Get SensorData
        # if count % (SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
        # camera_img = pybullet_util.get_camera_image_from_link(
        # robot, link_id['head'], 60., 2., 0.1, 10)
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)
        # Set up camera position and orientation relative to the robot
        

        # Set the camera's position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=-camera_pitch, cameraTargetPosition=camera_target_position)

        rf_height = pybullet_util.get_link_iso(robot, link_id['r_sole'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot, link_id['l_sole'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # ############################################################################################
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--nav_policy", type=str, default="bcrnn",
        #                 help="path for loading checkpoints, configuration and training logs. For example, --nav_policy=NAV_POLICY will load checkpoints at ./save/bc_checkpoints/NAV_POLICY.")
        # args = parser.parse_args()
        # nav_policy = args.nav_policy
        # PATH_SICRIPT    = os.path.dirname(os.path.realpath(__file__))
        # PATH_ROOT   = os.path.dirname(PATH_SICRIPT)
        # SUBPATH = yaml.load(open(os.path.join(PATH_ROOT, 'path.yaml')), Loader=yaml.FullLoader)
        # PATH_CHECKPOINT_BC = os.path.join(PATH_ROOT, SUBPATH['BC Checkpoint'])
        # nav_path = "{}/{}/models/model_best_training.pth".format(PATH_CHECKPOINT_BC, nav_policy)
        # eval_policy = policy_from_checkpoint(ckpt_path=nav_path)[0]

        # ##############################################################################################

        #==========================#   UPDATING FOR DEPTH SCANNING
        position, orientation = p.getBasePositionAndOrientation(robot)
        view_point, _ = p.multiplyTransforms(position, orientation,_view_agent['offset'], [0, 0, 0, 1])
        view_rpy = p.getEulerFromQuaternion(orientation)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition = view_point,
            distance = _view_agent['dist'],
            roll = RAD_TO_DEG * (view_rpy[0] + _view_agent['roll']),
            pitch = RAD_TO_DEG * (view_rpy[1] + _view_agent['pitch']),
            yaw = RAD_TO_DEG * (view_rpy[2] + _view_agent['yaw']),
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(_view_agent['width']) / _view_agent['height'],
            nearVal=_view_agent['near'],
            farVal=_view_agent['far'])
        (_, _, rgb, depth, _) = p.getCameraImage(
            width=_view_agent['width'],
            height=_view_agent['height'],
            renderer=p.ER_TINY_RENDERER,
            viewMatrix=view_matrix,
            shadow=0,
            projectionMatrix=proj_matrix)

        _pixels = {}
        _pixels['rgb'] = np.array(rgb)[:, :, 2::-1]
        _pixels['depth'] = np.array((1-depth)*255, dtype=np.uint8)
        rgbd = np.concatenate((_pixels['rgb'], np.sqrt(_pixels['depth'])[:, :, np.newaxis]), axis=2)/255.
        # rgbd = np.concatenate((self._pixels['rgb'], np.sqrt(self._pixels['depth'])[:, :, np.newaxis]), axis=2)/255.

        # p.resetDebugVisualizerCamera( cameraDistance = _view_fpv['dist'],
        #                                                         cameraTargetPosition = view_point,
        #                                                         cameraPitch = RAD_TO_DEG * _view_fpv['pitch'],
        #                                                         cameraYaw = RAD_TO_DEG * _view_fpv['yaw']
        #                                                         )

        _yaw = view_rpy[2]
        # print("\nyaw",_yaw)
        obs = {'rgbd': rgbd, 'yaw':_yaw, 'action': _nav_action}
        obs_dict = {
        "agentview_rgb": 255.*np.transpose(obs["rgbd"][..., :3], (2, 0, 1)),
        "agentview_depth": np.transpose(obs["rgbd"][..., 3:], (2, 0, 1)),
        "yaw": np.array([obs["yaw"]])
        }
        action = eval_policy(obs_dict)
        
        _nav_action = np.clip(action, [0, -1.0], [1., 1.0])
        yaw_robot_cmd = action[1]
        dist_robot_cmd = action[0]
        # angle = np.arctan2(_nav_action[0],_nav_action[1])
        print("\ndist robot command",dist_robot_cmd,"robot angle command",yaw_robot_cmd, "current yaw", _yaw)
        # Update Navigation Controller
        
        

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, '8'):
            interface.interrupt_logic.b_interrupt_button_eight = True
        elif pybullet_util.is_key_triggered(keys, '5'):
            interface.interrupt_logic.b_interrupt_button_five = True
        elif pybullet_util.is_key_triggered(keys, '4'):
            interface.interrupt_logic.b_interrupt_button_four = True
        elif pybullet_util.is_key_triggered(keys, '2'):
            interface.interrupt_logic.b_interrupt_button_two = True
        elif pybullet_util.is_key_triggered(keys, '6'):
            interface.interrupt_logic.b_interrupt_button_six = True
        elif pybullet_util.is_key_triggered(keys, '7'):
            interface.interrupt_logic.b_interrupt_button_seven = True
        elif pybullet_util.is_key_triggered(keys, '9'):
            interface.interrupt_logic.b_interrupt_button_nine = True

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Apply Trq
        pybullet_util.set_motor_trq(robot, joint_id, command)

        p.stepSimulation()

        # time.sleep(dt)
        t += dt
        count += 1
