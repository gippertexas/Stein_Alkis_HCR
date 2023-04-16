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
<<<<<<< Updated upstream
Filter = 0.667
nominalForward = 0.25
=======


>>>>>>> Stashed changes
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

def getRayFromTo(mouseX, mouseY):
  width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera(
  )
  camPos = [
      camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
      camTarget[2] - dist * camForward[2]
  ]
  farPlane = 10000
  rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
  lenFwd = math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] +
                     rayForward[2] * rayForward[2])
  invLen = farPlane * 1. / lenFwd
  rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
  rayFrom = camPos
  oneOverWidth = float(1) / float(width)
  oneOverHeight = float(1) / float(height)

  dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
  dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
  rayToCenter = [
      rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
  ]
  ortho = [
      -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] - float(mouseY) * dVer[0],
      -0.5 * horizon[1] + 0.5 * vertical[1] + float(mouseX) * dHor[1] - float(mouseY) * dVer[1],
      -0.5 * horizon[2] + 0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
  ]

  rayTo = [
      rayFrom[0] + rayForward[0] + ortho[0], rayFrom[1] + rayForward[1] + ortho[1],
      rayFrom[2] + rayForward[2] + ortho[2]
  ]
  lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
  alpha = math.atan(lenOrtho / farPlane)
  return rayFrom, rayTo, alpha

def get_point_cloud(width, height, view_matrix, proj_matrix):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    image_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    depth = image_arr[3]

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels1 = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels1 = pixels1[z < 0.99]
    pixels1[:, 2] = 2 * pixels1[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels1.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points

def visionAngle(_nav_action):
    position, orientation = p.getBasePositionAndOrientation(robot)
    print(position)
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
<<<<<<< Updated upstream
    turn_angle = np.rad2deg(np.arctan2(_nav_action[1],_nav_action[0]))
    print(action)
    filt = action[0]
    return(turn_angle_radian)
=======
    turn_angle_degrees = np.rad2deg(np.arctan2(_nav_action[1],_nav_action[0]))

    distance_in_front = p.rayTest(position, [2,0,0])
    left_side = [position[0]+1,1,.5]
    right_side = [position[0]+1,-1,.5]

    # print(distance_in_front)
    imgW = imgWidth
    imgH = imgHeight
    # print('height/width',imgH,'/',imgW)
    depth_img_buffer = np.reshape(depth, [imgW,imgH])
    # print(depth_img_buffer)
    pointCloud = np.empty([imgH, imgW, 4])
    projMatrix = np.array(proj_matrix).reshape([4,4], order = 'F')
    vieMatrix = np.array(view_matrix).reshape([4,4],order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projMatrix, vieMatrix))
    # count_ = 0
    # y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    # y *= -1.
    # x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    # h = np.ones_like(z)

    # pixels1 = np.stack([x, y, z, h], axis=1)
    # # filter out "infinite" depths
    # pixels1 = pixels1[z < 0.99]
    # pixels1[:, 2] = 2 * pixels1[:, 2] - 1

    # # turn pixels to world coordinates
    # points = np.matmul(tran_pix_world, pixels1.T).T
    # points /= points[:, 3: 4]
    # points = points[:, :3]

    for h in range(imgH):
        for w in range(imgW):
            x = (2*w - imgW)/imgW
            y = -(2*h - imgH)/imgH  # be careful！ deepth and its corresponding position
            z = 2*depth_img_buffer[w,h] - 1
            pixPos = np.array([x, y, z, 1])
            positions = np.matmul(tran_pix_world, pixPos)

            pointCloud[h,w,:] = positions / positions[3]
            # count_+=1
    distance_list=[5]
    for i in pointCloud:
        distance=np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
        if distance<distance_list[0]:
            distance_list.pop(0)
            distance_list.append(distance)
            

    # print('pointcloud',pointCloud.shape)
    # print(distance_list)
    # print(depth_img_buffer)
    return(turn_angle_radian, view_matrix, proj_matrix, imgW, imgH)
>>>>>>> Stashed changes
    # return {'errors':errors, 'linear':xyz_vel, 'angular':rpy_vel, 'position': xyz_pos, 'orientation': rpy_pos, 'nav_action': _nav_action}



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

<<<<<<< Updated upstream
    cabinetId = p.loadURDF(cwd + "/data1/assets/furnitures/cabinet_3/model.urdf", [1.5, 0.5, 0.])
=======
    cabinetId = p.loadURDF(cwd + "/data1/assets/furnitures/cabinet_3/model.urdf", [1.0, 0., 0.])
>>>>>>> Stashed changes
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
            view_mat = visionAngle(_nav_action)[1]
            proj_mat = visionAngle(_nav_action)[2]
            width = visionAngle(_nav_action)[3]
            height = visionAngle(_nav_action)[4]
            turn_angle = np.rad2deg(turn_angle_rad)
            turn_func = DCMTrajectoryManager(DCMPlanner(),taf_container.com_task,taf_container.pelvis_ori_task,_robot,"l_sole","r_sole")
            # WalkingConfig.NOMINAL_TURN_RADIANS = turn_angle_rad
            DCMTrajectoryManager.nominal_turn_radians = turn_angle_rad/3
<<<<<<< Updated upstream
            
            # DCMTrajectoryManager.nominal_forward_step = (distFilt/Filter)*nominalForward
            print(turn_angle)
            if(turn_angle<-8):
                turn_func.turn_right()
                interface.interrupt_logic.b_interrupt_button_nine = True
                degree_offset += turn_angle
=======
            print(turn_angle)
            # pointCloud = get_point_cloud(width, height, view_mat, proj_mat)
            # print(pointCloud)
            if(turn_angle<-8):
                turn_func.turn_right()
                interface.interrupt_logic.b_interrupt_button_nine = True
>>>>>>> Stashed changes
            if(turn_angle>8):
                turn_func.turn_left()
                interface.interrupt_logic.b_interrupt_button_seven = True
            else:
                filter = (8-np.abs(turn_angle))/8
                DCMTrajectoryManager.nominal_forward_step = filter*nominalForward
                turn_func.walk_forward()
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
