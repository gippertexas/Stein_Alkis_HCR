import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import pybullet as p
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import numpy as np
np.set_printoptions(precision=2)

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import pybullet_util
from util import util
from util import liegroup
from pynput import keyboard
from pynput.keyboard import Listener


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
    p.connect(p.DIRECT)
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

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, True)

    # Create Robot for Meshcat Visualization
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        cwd + "/robot_model/atlas/atlas.urdf", cwd + "/robot_model/atlas",
        pin.JointModelFreeFlyer())
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    viz.loadViewerModel()
    vis_q = pin.neutral(model)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    # Construct Interface
    interface = AtlasInterface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0

    time.sleep(1)

    def on_press(key):  # The function that's called when a key is pressed
        if hasattr(key, 'char'):
            if key.char == '8':
                print("8 is pressed")
                interface.interrupt_logic.b_interrupt_button_eight = True
            elif key.char == '5':
                print("5 is pressed")
                interface.interrupt_logic.b_interrupt_button_five = True
            elif key.char == '4':
                print("4 is pressed")
                interface.interrupt_logic.b_interrupt_button_four = True
            elif key.char == '2':
                print("2 is pressed")
                interface.interrupt_logic.b_interrupt_button_two = True
            elif key.char == '6':
                print("6 is pressed")
                interface.interrupt_logic.b_interrupt_button_six = True
            elif key.char == '7':
                print("7 is pressed")
                interface.interrupt_logic.b_interrupt_button_seven = True
            elif key.char == '9':
                print("9 is pressed")
                interface.interrupt_logic.b_interrupt_button_nine = True

    def on_release(key):  # The function that's called when a key is released
        pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while (1):

        # Get SensorData
        # if count % (SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
        # camera_img = pybullet_util.get_camera_image_from_link(
        # robot, link_id['head'], 60., 2., 0.1, 10)
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        rf_height = pybullet_util.get_link_iso(robot, link_id['r_sole'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot, link_id['l_sole'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Apply Trq
        pybullet_util.set_motor_trq(robot, joint_id, command)

        # Visualize via Meshcat
        vis_q[0:3] = sensor_data['base_joint_pos']
        vis_q[3:7] = sensor_data['base_joint_quat']
        for i, (k, v) in enumerate(sensor_data['joint_pos'].items()):
            vis_q[i + 7] = v
        viz.display(vis_q)

        # Save Image
        # if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
        # frame = pybullet_util.get_camera_image([1.2, 0.5, 1.], 2.0, 120,
        # -15, 0, 60., 1920, 1080,
        # 0.1, 100.)
        # frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
        # filename = video_dir + '/step%06d.jpg' % count
        # cv2.imwrite(filename, frame)

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1
