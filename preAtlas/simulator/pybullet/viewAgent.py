import pybullet as p
import numpy as np
import argparse
import yaml
import os
cwd = os.getcwd()

from robomimic.utils.file_utils import policy_from_checkpoint
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from config.atlas_config import PnCConfig
from util import pybullet_util
from config.atlas_config import SimConfig
from Environment_Setup import createRobot

DEG_TO_RAD = np.pi/180.
RAD_TO_DEG = 180./np.pi
scale = 33./13.



_view_agent = {'dist': 0.2,
                            'offset': np.array([0.45, 0, 0.]),
                            'roll' : 0. * DEG_TO_RAD,
                            'yaw' : -90. * DEG_TO_RAD,
                            'pitch': 0. * DEG_TO_RAD,
                            'width': 212,
                            'height': 120,
                            'near': 0.1,
                            'far': 100}

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

eval_policy = train_eval()

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

def visionAngle(_nav_action,robot):
    position, orientation = p.getBasePositionAndOrientation(robot)
    view_point, _ = p.multiplyTransforms(position, orientation,_view_agent['offset'], [0, 0, 0, 1])
    view_rpy = p.getEulerFromQuaternion(orientation)
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
    
    _yaw = view_rpy[2]
    # print("\nyaw",_yaw)
    obs = {'rgbd': rgbd, 'yaw':_yaw, 'action': _nav_action}
    obs_dict = {
    "agentview_rgb": 255.*np.transpose(obs["rgbd"][..., :3], (2, 0, 1)),
    "agentview_depth": np.transpose(obs["rgbd"][..., 3:], (2, 0, 1)),
    "yaw": np.array([obs["yaw"]])
    }
    
    action = eval_policy(obs_dict)
    # # # print(action)
    
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
    return(turn_angle_radian)
    # return {'errors':errors, 'linear':xyz_vel, 'angular':rpy_vel, 'position': xyz_pos, 'orientation': rpy_pos, 'nav_action': _nav_action}