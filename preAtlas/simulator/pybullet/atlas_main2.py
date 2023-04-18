import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import simulator.environments
import argparse
import yaml
from robomimic.utils.file_utils import policy_from_checkpoint
import torch
from pnc.wbc.manager.dcm_trajectory_manager import DCMTrajectoryManager
from pnc.planner.locomotion.dcm_planner.dcm_planner import DCMPlanner
from pnc.wbc.manager.floating_base_trajectory_manager import FloatingBaseTrajectoryManager
from pnc.wbc.basic_task import BasicTask
from config.atlas_config import PnCConfig

import pybullet as p
import numpy as np
np.set_printoptions(precision=2)

from pnc.atlas_pnc.atlas_task_force_container import AtlasTaskForceContainer
import abc
from pnc.interface import Interface
from pnc.control_architecture import ControlArchitecture
from config.atlas_config import WalkingState
from pnc.atlas_pnc.atlas_control_architecture import AtlasControlArchitecture
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider

from config.atlas_config import SimConfig
from config.atlas_config import WalkingConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import pybullet_util
from util import util
from util import liegroup
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from Environment_Setup import createRobot
DEG_TO_RAD = np.pi/180.
RAD_TO_DEG = 180./np.pi
SUBPATH_CONFIG = {  "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml",
                    "simulation": "simulation.yaml"}

scale = 33./13.
Filter = 0.667
nominalForward = 0.2

def train_eval():
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
    return eval_policy


def visionAngle(_nav_action):
    position, orientation = p.getBasePositionAndOrientation(robot)
    # print(orientation)
    view_point, _ = p.multiplyTransforms(position, orientation,_view_agent['offset'], [0, 0, 0, 1])
    view_rpy = p.getEulerFromQuaternion(orientation)
    # print('pitch',view_rpy[1])
    # if (t%50 == 0):
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
    (imgWidth, imgHeight, rgb, depth, _) = p.getCameraImage(
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
    
    _yaw = view_rpy[2]
    # print("\nyaw",_yaw)
    obs = {'rgbd': rgbd, 'yaw':_yaw, 'action': _nav_action}
    obs_dict = {
    "agentview_rgb": 255.*np.transpose(obs["rgbd"][..., :3], (2, 0, 1)),
    "agentview_depth": np.transpose(obs["rgbd"][..., 3:], (2, 0, 1)),
    "yaw": np.array([obs["yaw"]])
    }
    
    action = eval_policy(obs_dict)
    # # # # print(action)
    # print('yaw',_yaw)
    _nav_action = np.clip(action, [0, -1.0], [1., 1.0])
    target_xy=np.zeros(2)
    target_xy[0] = action[0]/scale
    target_yaw = action[1]/scale
    xyz_pos = position
    roll_pitch_yaw = p.getEulerFromQuaternion(orientation)
    linear_velocity = p.getBaseVelocity(robot)[0]
    angular_velo = p.getBaseVelocity(robot)[1]
    xyz_vel = TransformAngularVelocityToLocalFrame(linear_velocity,orientation)
    rpy_pos = roll_pitch_yaw
    rpy_vel = TransformAngularVelocityToLocalFrame(angular_velo, orientation)
    errors = np.concatenate(((scale * target_xy - xyz_vel[0:2]), scale * target_yaw -rpy_vel[2]), axis=None)
    turn_angle_radian = np.arctan2(_nav_action[1],_nav_action[0])
    turn_angle = np.rad2deg(np.arctan2(_nav_action[1],_nav_action[0]))
    # print(action)
    filt = action[0]
    print('angle from prelude',turn_angle_radian)
    pre_ang = turn_angle_radian
    
    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    projMatrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    viewMatrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(projMatrix, viewMatrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / imgHeight, -1:1:2 / imgWidth]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    # pixels = pixels[z < 0.99]
    # pixels[:, 2] = 2 * pixels[:, 2] - 1
    # print('pixels',pixels.shape)
    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    point_list = []
    for i in points:
        if i[2]<.5:
            point_list.append(i)
    points = point_list
    # print(len(points))
    yaw_pitch_roll = p.getEulerFromQuaternion(orientation)
    current_yaw = yaw_pitch_roll[2]
    point_list_dist = [np.sqrt((i[0]-position[0])**2+(i[1]-position[1])**2+(i[2]-position[2])**2) for i in points]
    point_list_angle = [np.arctan((i[1]-position[1])/(i[0]-position[0])) for i in points]
    count_ = 0
    ob_dic = {}
    # get obstacle dictionary for points that are less than 1.4m in distance
    # and clearance is less than .6, dictionary is organized as clearance:[distance to point, yaw angle to point]
    for i in point_list_dist:
        if .5< i <2.1:
            clearance = point_list_dist[count_]*np.sin(point_list_angle[count_] - current_yaw)
            if abs(clearance)<.75:
                ob_dic[clearance]=[point_list_dist[count_],point_list_angle[count_] - current_yaw]
        count_+=1
    # sort dictionary according to the clearance
    # from minimum value to highest value
    # get minimum value and find the desired yaw angle to get away from smallest clearance value
    # ob_dic = dict(sorted(ob_dic.items(), key=lambda item: abs(item[0])))
    ob_dic = dict(sorted(ob_dic.items(), key=lambda item: abs(item[1][1]), reverse = True))
    angle_des = 0
    turn_cm = "straight"
    if ob_dic:
        pt_dist = list(ob_dic.values())[0][0]
        pt_yaw = list(ob_dic.values())[0][1]
        pt_clear = list(ob_dic.keys())[0]
        print('pt distance',pt_dist)
        angle_des = np.arcsin(.6/pt_dist)-abs(pt_yaw)
        angle_des = angle_des*1.5
        if pt_clear<0:
            angle_des = abs(pre_ang)
        if pt_clear>0:
            angle_des =abs(pre_ang)*-1
        if pt_clear<0:
            turn_cm = "left"
        elif pt_clear>0:
            turn_cm="right"
        else:
            turn_cm="straight"


    # this stuff was used to learn from mistakes
    """
    # point_list_clearance_angle = [i-current_yaw for i in point_list_angle]
    # point_list_clearance_dist = [point_list_dist[i]*np.sin(point_list_clearance_angle[i]) for i in range(len(point_list_clearance_angle))]
    # for i in point_list_clearance_dist:
    # large_clearance_dic = {}
    
    
    # clearance_dic={}
    # count_=0
    # for i in point_list_angle:
        # clearance = point_list_dist[count_]*np.sin(i - current_yaw)
        # if abs(clearance)>1:
    #     clearance = point_list_dist[count_]*np.sin(i - current_yaw)
    #     # if abs(clearance)
    #     large_clearance_dic[clearance]=[point_list_dist[count_], i]
    #     count_+=1
    # # large_clearance_dic =dict(sorted(large_clearance_dic, key=lambda dict_key: large_clearance_dic[dict_key][1]))
    # # large_clearance_dic1 = dict(sorted(large_clearance_dic.items(), key=lambda item: abs(item[1][0]), reverse = True))
    # # print(large_clearance_dic)
    
    # # for key, val in large_clearance_dic.items():
    # large_clearance_dic = dict(sorted(large_clearance_dic.items(), key=lambda item: abs(item[1][0])))
    # clear_new = dict(sorted(large_clearance_dic.items(), key=lambda item: abs(item[0])))
    # distance_new = list(clear_new.values())[0][0]

    # ###########################################
    # ###########################################
    # count_=0
    # clearance_dic1={}
    # for i in point_list_angle:
    #     clearance = point_list_dist[count_]*np.sin(i - current_yaw)
    #     if .00001<abs(clearance)<1:
    #         clearance_dic[clearance]=[point_list_dist[count_], i]
    #         # clearance_angle = 
    #     if abs(clearance)>.6:
    #         clearance_dic1[clearance]=[point_list_dist[count_], i]
    #     count_+=1
    # # print(clearance_dic)
    # clear_dic={key:val for key, val in clearance_dic.items() if abs(val[1])<.4}
    # # print(clear_dic)
    # clear_dic_l = list(clear_dic.keys())
    # clear_dic = dict(sorted(clear_dic.items(), key=lambda item: abs(item[1][1])))
    # # clear_dic = 
    # large_clearance_dic1 = dict(sorted(clearance_dic1.items(), key=lambda item: abs(item[1][0]), reverse = True))

    # turn_angle_new = list(large_clearance_dic1.values())[0][1]
    # distance = list(large_clearance_dic1.values())[0][0]
    # # distance=list()
    # clearance_new = list(large_clearance_dic1.keys())[0]
    # des_clearance_angle=0
    # clearance_dic = {key:val for key, val in clearance_dic.items() if val[0]<1.4}
    # clearance_dic = dict(sorted(clearance_dic.items()))
    # # print(clearance_dic)
    # key_dic_list = list(clearance_dic.keys())
    # value_dic_list = list(clearance_dic.values())
    # if len(key_dic_list)>1:
    #     if key_dic_list[0]<0 and key_dic_list[len(key_dic_list)-1]<0:
    #         clearance_value = key_dic_list[len(key_dic_list)-1]
    #         des_clearance_angle = value_dic_list[len(value_dic_list)-1][1]-current_yaw
    #     elif key_dic_list[0]>0 and key_dic_list[len(key_dic_list)-1]>0:
    #         clearance_value = key_dic_list[0]
    #         des_clearance_angle = value_dic_list[0][1]-current_yaw
    # # obstacle_dic = {}
    # # for key, val in large_clearance_dic.items():
    # #     if abs(val[0])<1:
    # #         print(val)
    # #         if abs(val[0])<1.5:
    # #             print(val)
    # #             obstacle_dic[key] = val
    # # print(obstacle_dic)
    # # distances_obs_dic = list(obstacle_dic.values())


    # # for key, val in clearance_dic.items():
    # #      if val[0]<.6:
    # #           print(val[0])
    # # for key, val in clearance_dic.items():
    # #     print(val[0])
    
    # clearance_dic = dict(sorted(clearance_dic.items()))
    # # print(clearance_dic)
    # # dist_from_rob = 
    # min_value = min(point_list_dist)
    # min_index = point_list_dist.index(min_value)
    # angle_min_value = point_list_angle[min_index]

    # # if angle_min_value > 0:
    # #     clearance = min_value*np.sin(angle_min_value-current_yaw)
    # # if angle_min_value<0:

    # #     clearance = min_value*np.sin(angle_min_value+current_yaw)
    # # clearance = min_value*np.sin(angle_min_value - current_yaw)
    # clearance_angl = angle_min_value - current_yaw
    # clearance_value = min_value*np.sin(clearance_angl)
    # # print('clearance angle',clearance_angle, clearance_value)
    
    # desired_angle = np.arcsin(.5/min_value)-abs(clearance_angl)

    # # print('desired angle',desired_angle,'min value', min_value)
    # # if desired_angle>.6:
    # #     desired_angle=.6
    # # if clearance_value>0:
    # #     desired_angle*=-1
         
    # # print('min value',min_value,'angle min value',angle_min_value,'currentyaw',current_yaw,'angle to clearance',np.rad2deg(angle_min_value-current_yaw))
    # # print('clearance',clearance)
    # # valueForClearance = .6/min_value
    # # if angle_min_value<0:
    # #     valueForClearance*=(-1)
    # # desired_yaw = angle_min_value - np.arcsin(valueForClearance)
    
    # #     if key_dic_list[0]<0 and key_dic_list[len(key_dic_list)-1]<0:
    # #         clearance_value = key_dic_list[len(key_dic_list)-1]
    # #         des_clearance_angle = value_dic_list[len(value_dic_list)-1][1]-current_yaw
    # #     elif key_dic_list[0]>0 and key_dic_list[len(key_dic_list)-1]>0:
    # #         clearance_value = key_dic_list[0]
    # #         des_clearance_angle = value_dic_list[0][1]-current_yaw
    
    # # if des_clearance_angle!=0:
    # #     if clearance_value<0:

    # #         turn_angle_radian=abs(des_clearance_angle)
    # #     elif clearance_value>0:
    # #         turn_angle_radian = -1*abs(des_clearance_angle)
    # # print('turn angle radian',turn_angle_radian, clearance_value)
    # value_dic_list_big = list(large_clearance_dic1.values())
    # turn_angle_radian = value_dic_list_big[len(value_dic_list_big)-1][1]
    # clearances = list(large_clearance_dic.keys())
    # # print(len(clear_dic_l))
    # if len(clear_dic_l)>1:
    #     # print('yes')
    #     clearance_new=clearance_new
    # else:
    #     clearance_new = 0
        
    # print('turn angle radian',turn_angle_radian,clearance_value)
    """
    print('angle desired',angle_des)
    return(angle_des, turn_cm)



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
def TransformAngularVelocityToLocalFrame(angular_velocity,
                                           orientation):
    """Transform the angular velocity from world frame to robot's frame.

    Args:
      angular_velocity: Angular velocity of the robot in world frame.
      orientation: Orientation of the robot represented as a quaternion.

    Returns:
      angular velocity of based on the given orientation.
    """
    # Treat angular velocity as a position vector, then transform based on the
    # orientation given by dividing (or multiplying with inverse).
    # Get inverse quaternion assuming the vector is at 0,0,0 origin.
    _, orientation_inversed = p.invertTransform(
        [0, 0, 0], orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = p.multiplyTransforms(
        [0, 0, 0], orientation_inversed, angular_velocity,
        p.getQuaternionFromEuler([0, 0, 0]))
    return np.array(relative_velocity)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # # Environment Setup
    # p.connect(p.GUI)
    # p.resetDebugVisualizerCamera(cameraDistance=1.5,
    #                              cameraYaw=120,
    #                              cameraPitch=-30,
    #                              cameraTargetPosition=[1, 0.5, 1.5])
    # p.setGravity(0, 0, -9.8)
    # p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
    #                             numSubSteps=SimConfig.N_SUBSTEP)
    # # if SimConfig.VIDEO_RECORD:
    # # video_dir = 'video/atlas_pnc'
    # # if os.path.exists(video_dir):
    # # shutil.rmtree(video_dir)
    # # os.makedirs(video_dir)

    # # Create Robot, Ground
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
    #                    SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
    #                    SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    # planeId = p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    #     robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
    #     SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)
    
    # #######################################################################
    robot = createRobot()
    
    planeId = p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)
    FIELD_RANGE = [50, 3]
    FURNITURES = {}
    OBJECTS = {}
    assets = {}
    mass = 0
    setup_obstacles = {}
    p.changeVisualShape(planeId, -1, textureUniqueId=p.loadTexture("data1/textures/carpet.png"), rgbaColor=(0.8,0.7,0.8,1.0))
    setup_obstacles = simulator.environments.PerceptionWrapper._generate_obstacle_profiles(p,10,10,10,50,2)
    assets = simulator.environments.PerceptionWrapper._set_world(p,setup_obstacles)

    # chairId = p.loadURDF(cwd + "/data1/assets/furnitures/chair_1/model.urdf", [2.5, 0, 0])
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    #     robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
    #     SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    cabinetId = p.loadURDF(cwd + "/data1/assets/furnitures/cabinet_3/model.urdf", [.75, 0.2, 0.])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)
    
    cabinetId2 = p.loadURDF(cwd + "/data1/assets/furnitures/cabinet_3/model.urdf", [2, -1.2, 0.])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)
    
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
    eval_policy = train_eval()
    # ######################################################################################### (get obs)
    _view_agent = {'dist': 0.2, #originally 0.2
                            'offset': np.array([0.45, 0, 0.]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': 0. * DEG_TO_RAD,
                            'width': 212,
                            'height': 120,
                            'near': 0.1,
                            'far': 100}

    # _view_bird = {'dist': 2.0,
    #                         'offset': np.array([0.45, 0, 0]),
    #                         'roll' : 0. * DEG_TO_RAD,
    #                         'yaw' : -60. * DEG_TO_RAD,
    #                         'pitch': -30 * DEG_TO_RAD,
    #                         'width': 480,
    #                         'height': 360,
    #                         'near': 0.1,
    #                         'far': 100}

    # _view_fpv = {'dist': 1.5,
    #                         'offset': np.array([0.45, 0, 0]),
    #                         'roll' : 0. * DEG_TO_RAD,
    #                         'yaw' : -90. * DEG_TO_RAD,
    #                         'pitch': -30 * DEG_TO_RAD,
    #                         'width': 480,
    #                         'height': 360,
    #                         'near': 0.1,
    #                         'far': 100}
    

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
    _robot = PinocchioRobotSystem(
            cwd + "/robot_model/atlas/atlas.urdf",
            cwd + "/robot_model/atlas", False, PnCConfig.PRINT_ROBOT_INFO)
    ctrl_arch = AtlasControlArchitecture(_robot)
    degree_offset = 0

    
   
    while (1):
        
        # # Get SensorData
        # # if count % (SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
        # # camera_img = pybullet_util.get_camera_image_from_link(
        # # robot, link_id['head'], 60., 2., 0.1, 10)
        # # print('robot',robot)
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)
        # # Set up camera position and orientation relative to the robot
        

        # # Set the camera's position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=-camera_pitch, cameraTargetPosition=camera_target_position)

        rf_height = pybullet_util.get_link_iso(robot, link_id['r_sole'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot, link_id['l_sole'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False
        

        # ############################################################################################
        

        # ##############################################################################################

        #==========================#   UPDATING FOR DEPTH SCANNING
        
        # yaw_robot_cmd = action[1]
        # dist_robot_cmd = action[0]
        # turn_angle_radian = np.arctan2(_nav_action[1],_nav_action[0])
        # turn_angle = np.rad2deg(np.arctan2(_nav_action[1],_nav_action[0]))
        # # print(turn_angle)
        # # from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
        # # _robot = PinocchioRobotSystem(
        # #     cwd + "/robot_model/atlas/atlas.urdf",
        # #     cwd + "/robot_model/atlas", False, PnCConfig.PRINT_ROBOT_INFO)
        # # print('robot',_robot)
        taf_container = AtlasTaskForceContainer(_robot)
        # # _com_task = BasicTask(robot, "COM", 3, 'com', PnCConfig.SAVE_DATA)
        # # _pelvis_ori_task = BasicTask(robot, "LINK_ORI", 3, "pelvis_com",
        # #                               PnCConfig.SAVE_DATA)
        
        # # # print(turn_angle)
        # ctrl_arch = AtlasControlArchitecture(_robot)
        # # if ctrl_arch.state == WalkingState.BALANCE:
        # #     print("hell yeah boi")
        



        

        # # Get Keyboard Event
        # keys = p.getKeyboardEvents()
        # if pybullet_util.is_key_triggered(keys, '8'):
        #     interface.interrupt_logic.b_interrupt_button_eight = True #walk forward
        # elif pybullet_util.is_key_triggered(keys, '5'):
        #     interface.interrupt_logic.b_interrupt_button_five = True #walk in place
        # elif pybullet_util.is_key_triggered(keys, '4'):
        #     interface.interrupt_logic.b_interrupt_button_four = True #walk left
        # elif pybullet_util.is_key_triggered(keys, '2'):
        #     interface.interrupt_logic.b_interrupt_button_two = True #walk backward
        # elif pybullet_util.is_key_triggered(keys, '6'):
        #     interface.interrupt_logic.b_interrupt_button_six = True #walk right
        # elif pybullet_util.is_key_triggered(keys, '7'):
        #     interface.interrupt_logic.b_interrupt_button_seven = True #turn left
        # elif pybullet_util.is_key_triggered(keys, '9'):
        #     interface.interrupt_logic.b_interrupt_button_nine = True #turn right
        # if the state is balance then continue to next command
        if AtlasStateProvider(ctrl_arch)._state == 1:
            turn_angle_rad = visionAngle(_nav_action)[0]
            turn_cm = visionAngle(_nav_action)[1]
            distance = .5
            steps = math.floor(distance/.2)
            DCMTrajectoryManager.nominal_forward_step = distance/steps
            DCMTrajectoryManager.nominal_forward_number=steps
            # pointcloud = visionAngle(_nav_action)[1]
            # print(pointcloud)
            # view_mat = visionAngle(_nav_action)[1]
            # proj_mat = visionAngle(_nav_action)[2]
            # width = visionAngle(_nav_action)[3]
            # height = visionAngle(_nav_action)[4]
            # turn_angle = np.rad2deg(turn_angle_rad)
            # turn_func = DCMTrajectoryManager(DCMPlanner(),taf_container.com_task,taf_container.pelvis_ori_task,_robot,"l_sole","r_sole")
            # WalkingConfig.NOMINAL_TURN_RADIANS = turn_angle_rad
            DCMTrajectoryManager.nominal_turn_radians = turn_angle_rad
            # Stand = True
            
            # DCMTrajectoryManager.nominal_forward_step = (distFilt/Filter)*nominalForward
            # print('turn angle in radians',turn_angle_rad)
            # pointCloud = get_point_cloud(width, height, view_mat, proj_mat)
            # print(pointCloud)
            # clearance_value = visionAngle(_nav_action)[1]
            # turn_func.turn_right()
            # interface.interrupt_logic.b_interrupt_button_nine = True
            if turn_cm == "right":
                # if turn_angle_rad>0:
                #     turn_angle_rad*=-1
                # turn_func.turn_right()
                interface.interrupt_logic.b_interrupt_button_nine = True
            elif turn_cm == "left":
                # print('turn agle in loop',turn_angle_rad)
                # if turn_angle_rad<0:
                #     turn_angle_rad*=-1
                # print('turn agle in loop',turn_angle_rad)
                # turn_func.turn_left()
                interface.interrupt_logic.b_interrupt_button_seven = True
                
                # filter = (8-np.abs(turn_angle))/8
                # DCMTrajectoryManager.nominal_forward_step = nominalForward
        # turn_func.walk_forward()
            elif turn_cm == "straight":
                interface.interrupt_logic.b_interrupt_button_eight = True
                
         # # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data)) # this is what makes it really slow
        # print('comand',command)

        if SimConfig.PRINT_TIME:
                    end_time = time.time()
                #     print("ctrl computation time: ", end_time - start_time)

        # # # Apply Trq
        pybullet_util.set_motor_trq(robot, joint_id, command)
        
        p.stepSimulation()    
        t += dt
        # print(t)
        count += 1
        # break
