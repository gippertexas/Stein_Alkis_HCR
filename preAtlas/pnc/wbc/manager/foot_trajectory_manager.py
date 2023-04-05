import copy

import numpy as np
from scipy.spatial.transform import Rotation as R

from pnc.planner.locomotion.dcm_planner.footstep import Footstep, interpolate
from util import util
from util import interpolation


class FootTrajectoryManager(object):
    """
    Foot SE(3) Trajectory Manager
    -----------------------------
    Usage:
        use_current or
        initialize_swing_foot_trajectory --> update_swing_foot_desired
    """
    def __init__(self, pos_task, ori_task, robot):
        self._pos_task = pos_task
        self._ori_task = ori_task
        self._robot = robot

        self._swing_start_time = 0.
        self._swing_duration = 0.

        self._swing_init_foot = Footstep()
        self._swing_mid_foot = Footstep()
        self._swing_land_foot = Footstep()

        self._pos_traj_init_to_mid = None
        self._pos_traj_mid_to_end = None
        self._quat_hermite_curve = None

        assert self._pos_task.target_id == self._ori_task.target_id
        self._target_id = self._pos_task.target_id

        # Attribute
        self._swing_height = 0.05

    def use_current(self):
        """
        Problem #1
        ----------
        Set foot position and orientation task through update_desired method.
        You would properly set foot_pos_des, foot_lin_vel_des for self._pos_task
        and set foot_rot_des, foot_ang_vel_des, for self._ori_task.
        Note that this use_current method is to set 0 acceleration to enforce
        unilateral constraint in the following pd control framework:
        xddot = des_x_acc + kp(des_x - x) + kd(des_xdot - xdot)

        Parameters to set
        -----------------
        foot_pos_des : 3d np array representing des foot pos
        foot_lin_vel_des : 3d np array representing des foot lin vel
        foot_rot_des : 4d np array representing des foot quat (scalar last repr)
        foot_ang_vel_des : 3d np array representing des foot ang vel

        Note
        ----
        You may need to use a function util.rot_to_quat to convert rotation
        matrix to quaternion.
        You may need to query the foot position and orientation of the
        self._target_id via self._robot.
        """
        POS = self._robot.get_link_iso(self._target_id)
        # RR = self._robot.get_link_iso("r_sole")
        # LL = self._robot.get_link_iso("l_sole")
        # print('R_foot:',RR,'L_foot:',LL)
        Px = POS[0,3]
        Py = POS[1,3]
        Pz = POS[2,3]
        ROT = np.array(POS[0:3,0:3])
        #ROT = np.linalg.inv(ROT)
        foot_pos_des = np.array([Px, Py, Pz])
        VEL = self._robot.get_link_vel(self._target_id)
        foot_lin_vel_des = np.array([VEL[3],VEL[4],VEL[5]])
        #foot_lin_vel_des = np.zeros(3)
        

        

        self._pos_task.update_desired(foot_pos_des, foot_lin_vel_des,np.zeros(3))
        foot_rot_des = util.rot_to_quat(ROT)
        #foot_rot_des = np.zeros(4)
        foot_ang_vel_des = np.array([VEL[0],VEL[1],VEL[2]])
        #foot_ang_vel_des = np.zeros(3)
        self._ori_task.update_desired(foot_rot_des, foot_ang_vel_des,
                                      np.zeros(3))

    def initialize_swing_foot_trajectory(self, start_time, swing_duration,
                                         landing_foot):
        self._swing_start_time = start_time
        self._swing_duration = swing_duration
        self._swing_land_foot = copy.deepcopy(landing_foot)

        self._swing_init_foot.iso = np.copy(
            self._robot.get_link_iso(self._target_id))
        self._swing_init_foot.side = landing_foot.side
        self._swing_mid_foot = interpolate(self._swing_init_foot, landing_foot,
                                           0.5)

        # compute midfoot boundary conditions
        mid_swing_local_foot_pos = np.array([0., 0., self._swing_height])
        mid_swing_pos = self._swing_mid_foot.pos + np.dot(
            self._swing_mid_foot.rot, mid_swing_local_foot_pos)
        mid_swing_vel = (self._swing_land_foot.pos -
                         self._swing_init_foot.pos) / self._swing_duration

        # construct trajectories
        self._pos_traj_init_to_mid = interpolation.HermiteCurveVec(
            self._swing_init_foot.pos, np.zeros(3), mid_swing_pos,
            mid_swing_vel)
        self._pos_traj_mid_to_end = interpolation.HermiteCurveVec(
            mid_swing_pos, mid_swing_vel, self._swing_land_foot.pos,
            np.zeros(3))
        self._quat_hermite_curve = interpolation.HermiteCurveQuat(
            self._swing_init_foot.quat, np.zeros(3),
            self._swing_land_foot.quat, np.zeros(3))
           

    def update_swing_foot_desired(self, curr_time):
        """
        Problem #2
        ----------
        Set foot position and orientation task through update_desired method.
        You would properly set foot_pos_des, foot_lin_vel_des, foot_lin_acc_des
        for self._pos_task and set foot_quat_des, foot_ang_vel_des,
        foot_ang_acc_des,for self._ori_task. Note that this
        update_swing_foot_desired method is to evalute foot trajectory at
        curr_time and set desired values for the tasks.

        Parameters to set
        -----------------
        foot_pos_des : 3d np array representing des foot pos
        foot_lin_vel_des : 3d np array representing des foot lin vel
        foot_lin_acc_des : 3d np array representing des foot lin acc
        foot_quat_des : 4d np array representing des foot quat (scalar last repr)
        foot_ang_vel_des : 3d np array representing des foot ang vel
        foot_ang_acc_des : 3d np array representing des foot lin acc

        Parameters to use
        -----------------
        self._swing_start_time : swing start time
        self._swing_duration : swing duration
        curr_time : current time querying desired values

        Note
        ----
        You first may need to compute a progression variable s given
        self._swing_duration, self._swing_start_time, and curr_time.
        Once you compute the progression variable use APIs evaluate,
        evaluate_first_derivative, evaluate_second_derivative in HermiteCurveVec
        class and evaluate, evaluate_ang_vel, evaluate_ang_acc in
        HermiteCurveQuat class to query desired values.
        """
        #FootTrajectoryManager.initialize_swing_foot_trajectory(self._swing_start_time,self._swing_duration,self._target_id)

        # Calculate the time elapsed since the start of the swing period

        t = np.clip(curr_time, self._swing_start_time,
        self._swing_start_time + self._swing_duration)
        time_since_swing_start = t - self._swing_start_time
        s = time_since_swing_start / self._swing_duration



        sinPos = interpolation.smooth_changing(0,1,1,s)
        sinVel = interpolation.smooth_changing_vel(0,1,1,s)
        sinAcc = interpolation.smooth_changing_acc(0,1,1,s)


        
        if(s<=0.5):
            foot_pos_des = self._pos_traj_init_to_mid.evaluate(2*s)
            foot_lin_vel_des = self._pos_traj_init_to_mid.evaluate_first_derivative(2*s)
            foot_lin_acc_des = self._pos_traj_init_to_mid.evaluate_second_derivative(2*s)
        else:
            x=(s-.5)*2
            foot_pos_des = self._pos_traj_mid_to_end.evaluate(x)
            foot_lin_vel_des = self._pos_traj_mid_to_end.evaluate_first_derivative(x)
            foot_lin_acc_des = self._pos_traj_mid_to_end.evaluate_second_derivative(x)

        self._pos_task.update_desired(foot_pos_des, foot_lin_vel_des,
                                      foot_lin_acc_des)

        foot_quat_des = self._quat_hermite_curve.evaluate(s)
        foot_ang_vel_des = self._quat_hermite_curve.evaluate_ang_vel(s)
        foot_ang_acc_des = self._quat_hermite_curve.evaluate_ang_acc(s)

        self._ori_task.update_desired(foot_quat_des, foot_ang_vel_des,
                                      foot_ang_acc_des)

    @property
    def swing_height(self):
        return self._swing_height

    @swing_height.setter
    def swing_height(self, val):
        self._swing_height = val
