import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import math
import pickle

import numpy as np

from pnc.planner.locomotion.dcm_planner.footstep import Footstep
from pnc.planner.locomotion.dcm_planner.dcm_planner import DCMPlanner


def save_trajectory(lf_stance, rf_stance, footstep_list, dcm_planner):
    t_start = dcm_planner.t_start
    t_end = t_start + dcm_planner.t_end
    t_step = 0.01
    n_eval = math.floor((t_end - t_start) / t_step)

    data = dict()

    # Temporal Info
    data["temporal_parameters"] = dict()
    data["temporal_parameters"]["initial_time"] = t_start
    data["temporal_parameters"]["final_time"] = t_end
    data["temporal_parameters"]["time_step"] = t_step
    data["temporal_parameters"]["t_ds"] = dcm_planner.t_ds
    data["temporal_parameters"]["t_ss"] = dcm_planner.t_ss
    data["temporal_parameters"]["t_transfer"] = dcm_planner.t_transfer

    # Contact Info
    data["contact"] = dict()
    data["contact"]["curr_right_foot"] = dict()
    data["contact"]["curr_left_foot"] = dict()
    data["contact"]["right_foot"] = dict()
    data["contact"]["left_foot"] = dict()
    data["contact"]["curr_right_foot"]["pos"] = np.copy(rf_stance.pos)
    data["contact"]["curr_right_foot"]["ori"] = np.copy(rf_stance.quat)
    data["contact"]["curr_left_foot"]["pos"] = np.copy(lf_stance.pos)
    data["contact"]["curr_left_foot"]["ori"] = np.copy(lf_stance.quat)

    rfoot_pos, rfoot_quat, lfoot_pos, lfoot_quat = [], [], [], []
    for i in range(len(footstep_list)):
        if footstep_list[i].side == Footstep.RIGHT_SIDE:
            rfoot_pos.append(footstep_list[i].pos)
            rfoot_quat.append(footstep_list[i].quat)
        else:
            lfoot_pos.append(footstep_list[i].pos)
            lfoot_quat.append(footstep_list[i].quat)
    data["contact"]["right_foot"]["pos"] = rfoot_pos
    data["contact"]["right_foot"]["ori"] = rfoot_quat
    data["contact"]["left_foot"]["pos"] = lfoot_pos
    data["contact"]["left_foot"]["ori"] = lfoot_quat

    # Ref Trajectory
    com_pos_ref = np.zeros((n_eval, 3))
    com_vel_ref = np.zeros((n_eval, 3))
    base_ori_ref = np.zeros((n_eval, 4))
    t_traj = np.zeros((n_eval, 1))

    t = t_start
    for i in range(n_eval):
        t_traj[i, 0] = t
        com_pos_ref[i, :] = dcm_planner.compute_reference_com_pos(t)
        com_vel_ref[i, :] = dcm_planner.compute_reference_com_vel(t)
        base_ori_ref[i, :], _, _ = dcm_planner.compute_reference_base_ori(t)
        t += t_step

    data["reference"] = dict()
    data["reference"]["com_pos"] = com_pos_ref
    data["reference"]["com_vel"] = com_vel_ref
    data["reference"]["base_ori"] = base_ori_ref
    data["reference"]["time"] = t_traj

    if not os.path.exists('data'):
        os.makedirs('data')

    file = open("data/dcm_planning.pkl", 'ab')
    pickle.dump(data, file)
    file.close()


## =============================================================================
## Main
## =============================================================================

initial_dcm = np.array([0., 0., 0.765])

dcm_planner = DCMPlanner()
dcm_planner.t_transfer = 0.
dcm_planner.t_ds = 0.45
dcm_planner.t_ss = 0.75
dcm_planner.percentage_settle = 0.9
dcm_planner.alpha_ds = 0.5
dcm_planner.robot_mass = 100.
dcm_planner.z_vrp = 0.765

lf_stance = Footstep()
lf_stance.pos = np.array([0., 0.11, 0.])
lf_stance.quat = np.array([0., 0., 0., 1.])
lf_stance.side = Footstep.LEFT_SIDE

rf_stance = Footstep()
rf_stance.pos = np.array([0., -0.11, 0.])
rf_stance.quat = np.array([0., 0., 0., 1.])
rf_stance.side = Footstep.RIGHT_SIDE

first_step = Footstep()
first_step.pos = np.array([0.15, 0.11, 0.])
first_step.quat = np.array([0., 0., 0., 1.])
first_step.side = Footstep.LEFT_SIDE

second_step = Footstep()
second_step.pos = np.array([0.30, -0.11, 0.])
second_step.quat = np.array([0., 0., 0., 1.])
second_step.side = Footstep.RIGHT_SIDE

third_step = Footstep()
third_step.pos = np.array([0.30, 0.11, 0.])
third_step.quat = np.array([0., 0., 0., 1.])
third_step.side = Footstep.LEFT_SIDE

input_footstep_list = [first_step, second_step, third_step]

dcm_planner.initialize(input_footstep_list, lf_stance, rf_stance, initial_dcm,
                       np.zeros(3))

save_trajectory(lf_stance, rf_stance, input_footstep_list, dcm_planner)
