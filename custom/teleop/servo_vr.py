#!/home/vision/anaconda3/envs/servo/bin/python

import os
import time
import cv2
import rospy
import signal
import numpy as np
import spatialmath.base as smb
import functools
import atexit

from api.cobot import *
from rb import *
from roboticstoolbox import models
from spatialmath import SE3
from tf2_msgs.msg import TFMessage

def rotation_safety_lock(delta_rot_rad, threshold_deg=178.0):
    global angle_lock
    global locked_value
    max_rad = np.deg2rad(threshold_deg)
    min_rad = -max_rad
    margin = np.deg2rad(1.0)
    delta_rot_out = delta_rot_rad.copy()

    for i in range(3):
        raw = delta_rot_rad[i]

        if angle_lock[0][i]:

            if max_rad - margin > raw > 0:
                angle_lock[0][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue

        elif angle_lock[1][i]:
            if min_rad + margin < raw < 0:
                angle_lock[1][i] = False
            else:
                delta_rot_out[i] = locked_value[i]
                continue

        if raw >= max_rad:
            angle_lock[0][i] = True
            locked_value[i] = max_rad
            delta_rot_out[i] = max_rad

        elif raw <= min_rad:
            angle_lock[1][i] = True
            locked_value[i] = min_rad
            delta_rot_out[i] = min_rad

    return delta_rot_out

def clip_pose(pose, xlim, ylim, zlim):
    t = pose.t.copy()
    t[0] = np.clip(t[0], xlim[0], xlim[1])
    t[1] = np.clip(t[1], ylim[0], ylim[1])
    t[2] = np.clip(t[2], zlim[0], zlim[1])
    return SE3.Rt(pose.R, t)

def vr_pose_callback(msg):
    global T_station2track
    for transform in msg.transforms:
        if transform.child_frame_id == TARGET_FRAME:
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            quat = [rot.w, rot.x, rot.y, rot.z]
            R = smb.q2r(quat)
            # print("pos:", pos)
            # print("R:", R)
            T_station2track = SE3.Rt(R, pos)

def linear_motion_with_target(robot, current_jnt, target_pose, step_time=0.01, acc_limit=40.0):
    global next_jnt
    current_pose = robot.fkine(current_jnt)
    pose_error = current_pose.inv() * target_pose

    err_pos = target_pose.t - current_pose.t
    err_rot_ee = smb.tr2rpy(pose_error.R, unit='rad')
    err_rot_base = current_pose.R @ err_rot_ee
    # print(f"err_err_rot_eerot:{err_rot_ee}")
    # print(f"err_rot_base:{err_rot_base}")
    # print(f"err_pos: {err_pos}\n")
    err_6d = np.concatenate((err_pos, err_rot_base))

    J = robot.jacob0(current_jnt)
    dq = np.linalg.pinv(J) @ err_6d

    if np.linalg.norm(dq[:3]) > acc_limit:
        dq[:3] *= acc_limit / np.linalg.norm(dq[:3])

    next_jnt = current_jnt + dq * 0.5
    ServoJ(next_jnt * 180 / np.pi)

def ServoJ(joint_deg, time1=0.002, time2=0.1, gain=0.02, lpf_gain=0.2):
    msg = f"move_servo_j(jnt[{','.join(f'{j:.3f}' for j in joint_deg)}],{time1},{time2},{gain},{lpf_gain})"
    SendCOMMAND(msg, CMD_TYPE.MOVE)

def _close_log():
    _log_file.close()
atexit.register(_close_log)

def log_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_tuple = func(*args, **kwargs)
        line = ','.join(str(x) for x in (time.time(), *data_tuple)) + '\n'
        _log_file.write(line)
        _log_file.flush()
        return data_tuple
    return wrapper

@log_data
def make_log_tuple(delta_vr_pos, delta_vr_rot, current_jnt, current_pose, next_jnt):
    # delta_vr_pos: np.array([dx,dy,dz])
    # delta_vr_rot: np.array([rx,ry,rz])
    # current_jnt: np.array([6])
    # current_pose: SE3
    # next_jnt: np.array([6])
    tcp_p = current_pose.t
    tcp_r = smb.tr2rpy(current_pose.R, unit='rad')
    return (
        delta_vr_pos[0], delta_vr_pos[1], delta_vr_pos[2],
        delta_vr_rot[0], delta_vr_rot[1], delta_vr_rot[2],

        *current_jnt.tolist(),

        tcp_p[0], tcp_p[1], tcp_p[2],
        tcp_r[0], tcp_r[1], tcp_r[2],

        *next_jnt.tolist()
    )

if __name__ == "__main__":

    log_path = os.path.join(os.getcwd(), 'experiment_data.txt')
    _log_file = open(log_path, 'w')
    _log_file.write(
        'timestamp,' +
        'vr_dx,vr_dy,vr_dz,vr_rx,vr_ry,vr_rz,' +
        'j0,j1,j2,j3,j4,j5,' +
        'tcp_x,tcp_y,tcp_z,tcp_rx,tcp_ry,tcp_rz,' +
        'next_j0,next_j1,next_j2,next_j3,next_j4,next_j5\n'
    )

    ToCB("192.168.111.50")
    robot = RB10()
    rospy.init_node("simple_calib")
    rospy.Subscriber("/tf", TFMessage, vr_pose_callback)
    TARGET_FRAME = "tracker_LHR_FA6FA4D9"

    CobotInit()
    mode = input("real? (y/n): ")
    if mode == 'y':
        SetProgramMode(PG_MODE.REAL)
    else:
        SetProgramMode(PG_MODE.SIMULATION)

    time.sleep(1)

    # Insert Calibrated Result Here...
    T_base2station = SE3.CopyFrom(
                np.array([[  -0.5446    ,0.01879 , -0.8385   ,-1.281      ],
                          [  -0.8386   ,-0.02055,   0.5443  , -0.7035     ],
                          [  -0.007003 , 0.9996 ,   0.02695 ,  1.204      ],
                          [   0        , 0      ,   0       ,  1  ]]), 
                check=False
                )

    Kp_pos = 0.8
    Kp_rot = 0.6

    T_tracker_init = None
    T_station2track = None
    init_pose = None

    next_jnt = None

    angle_lock = np.array([[False, False, False], # from pos->neg
                          [False, False, False]]) # from neg->pos

    locked_value = np.zeros(3)

    # define your workspace in meters (Robot base frame)
    X_MIN, X_MAX = -0.48, 0.35
    Y_MIN, Y_MAX = -0.73, -0.47
    Z_MIN, Z_MAX = 0.095, 0.81

    cv2.namedWindow('kill button')

    while not rospy.is_shutdown():

        if T_station2track is not None:

            current_jnt = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0
            current_pose = robot.fkine(current_jnt)
            if T_tracker_init is None:
                T_tracker_init = T_station2track  # store initial pose
                print("Initial marker pose stored.")

            if init_pose is None:
                init_pose = current_pose

            # Compute relative motion from initial marker pose in station frame
            T_rel = T_tracker_init.inv() * T_station2track
            # T_rel[-1] = 0.0
            delta_vr_pos = T_rel.t
            delta_vr_rot = smb.tr2rpy(T_rel.R, unit='rad')
            # print(f"Delta in station frame BEFORE(rot):\n{delta_vr_rot}")

            delta_vr_rot = rotation_safety_lock(delta_vr_rot)
            # delta_vr_rot = clip_rotation_vector(delta_vr_rot)
            # print(f"Delta in station frame (pos):\n{delta_vr_pos}")
            # print(f"Delta in station frame AFTER(rot):\n{delta_vr_rot}\n")

            # Convert to base frame: cam → end-effector → base
            R_base2station = T_base2station.R
            R_station2tracker = T_station2track.R
            R_tracker_init = T_tracker_init.R
            R_end2base = current_pose.inv().R
            R_end2base_init = init_pose.inv().R

            # delta_pos_base = R_base2station @ R_station2tracker @ delta_vr_pos
            # delta_pos_base = R_base2station @ R_tracker_init @ delta_vr_pos
            delta_pos_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_pos

            # delta_rot_base = R_base2station @ R_tracker_init @ delta_vr_rot  # rotational part treated as vector
            # delta_rot_base = R_base2station @ R_station2tracker @ delta_vr_rot  # rotational part treated as vector
            delta_rot_end = R_end2base_init @ R_base2station @ R_tracker_init @ delta_vr_rot  # rotational part treated as vector

            scaled_delta_pos = Kp_pos * delta_pos_end
            scaled_delta_rot = Kp_rot * np.array([delta_rot_end[0], delta_rot_end[1], delta_rot_end[2]])
            # scaled_delta_rot = Kp_rot * np.array([0 ,0 ,0])  # only Z-axis rotation

            # print(f"Delta in base frame (pos):\n{scaled_delta_pos}")
            # print(f"Delta in base frame (rot):\n{scaled_delta_rot}\n")

            # target_pose = SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit='rad', order='xyz') * init_pose
            # target_pose = SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit='rad') * init_pose
            target_pose = init_pose * SE3(scaled_delta_pos) * SE3.RPY(*scaled_delta_rot, unit='rad')
            ## or 
            # target_pose = init_pose * SE3.RPY(*scaled_delta_rot, unit='rad') * SE3(scaled_delta_pos)
            target_pose = clip_pose(target_pose, (X_MIN, X_MAX), (Y_MIN, Y_MAX), (Z_MIN, Z_MAX))

            # print(f"target_pose: \n{target_pose}")

            linear_motion_with_target(robot, current_jnt, target_pose)

            # make_log_tuple(delta_vr_pos, delta_vr_rot, current_jnt, current_pose, next_jnt)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("EXITING THE PROGRAM...")
            pid = os.getpid()
            os.kill(pid, signal.SIGKILL)
            break
