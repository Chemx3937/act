#!/usr/bin/env python3

# 수정 List:
# 1. action에서 Leader, 그리퍼의 pos 받게 하기 <- Joint Copy 방법 설치되면 하기
# 2. Image shape: 480,640,3
# 3. 초기위치 설정 필요 <- 해보고 성능 안좋으면 추가하기

import sys
sys.path.append('/home/vision/catkin_ws/src/robotory_rb10_rt/scripts')
import h5py
import rospy
import numpy as np
from teleop_data.msg import OnRobotRGOutput
from pynput import keyboard
from api.cobot import *
from rb import *
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
import time
from datetime import datetime
import os


def init_buffer():
    return {
        'observations': {
            'images': {
                'cam_high': [],
                'cam_low': [],
                # 'cam_left_wrist': [],
                # 'cam_right_wrist': []
            },
            'qpos': [],
            # 'qvel': [],
            # 'q_effort' : [],
        },
        'action': []
    }

def save_to_hdf5(buffer, i=None):
    today = datetime.now().strftime('%m%d')  # '0616' 형식
    base_dir = '/home/vision/catkin_ws/src/teleop_data/act_data'

    if i is None:
        # 자동 인덱스 결정: 같은 날짜의 기존 파일 개수 세기
        existing = [f for f in os.listdir(base_dir) if f.startswith(f'tele_data_{today}_') and f.endswith('.hdf5')]
        i = len(existing)

    filename = f'tele_data_{today}_Episode{i}.hdf5'
    save_path = os.path.join(base_dir, filename)

    with h5py.File(save_path, 'w') as f:
        f.attrs['sim'] = False
        obs = f.create_group('observations')
        imgs = obs.create_group('images')

        N = len(buffer['action'])

        # Create image datasets
        for cam in buffer['observations']['images']:
            imgs.create_dataset(cam, data=np.array(buffer['observations']['images'][cam], dtype=np.uint8))

        # qpos, qvel, action
        obs.create_dataset('qpos', data=np.array(buffer['observations']['qpos'], dtype=np.float64))
        # obs.create_dataset('qvel', data=np.array(buffer['observations']['qvel'], dtype=np.float64))
        f.create_dataset('action', data=np.array(buffer['action'], dtype=np.float64))

        print(f"[HDF5] Saved {N} timesteps to {save_path}")
        return i

def gripper_callback(msg):
    global latest_gripper_qpos
    latest_gripper_qpos = [msg.rGWD]

def on_press(key):
    global recording, terminal
    try:
        if key.char == 's':
            recording = True
            print("Start recording")
        elif key.char == 'q':
            recording = False
            print("Stop recording")
        elif key.char == 't':
            terminal = True
    except AttributeError:
        pass

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    if len(serials) < 2:
        raise RuntimeError("2개 이상의 Realsense 카메라가 연결되어 있어야 합니다.")
    print("Detected serials:", serials)
    return serials

def main():
    # global i
    # i=0

    serials = get_device_serials()
    serial_d435 = serials[0]
    serial_d405 = serials[1]

    pipeline0 = rs.pipeline()
    config0 = rs.config()
    config0.enable_device(serial_d435)
    config0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline0.start(config0)

    pipeline1 = rs.pipeline()
    config1 = rs.config()
    config1.enable_device(serial_d405)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline1.start(config1)

    CROP_SIZE = 480
    ToCB("192.168.111.50")
    robot = RB10()
    CobotInit()

    global terminal, recording, latest_gripper_qpos
    recording = False
    terminal = False
    latest_gripper_qpos = None

    rospy.init_node("hdf_maker2")
    rospy.Subscriber("/OnRobotRGOutput", OnRobotRGOutput, gripper_callback)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    buffer = init_buffer()
    rate = rospy.Rate(20) # rate 왜 20Hz인지 물어보기 # act는 50Hz

    # pose, gripper 값 정규화위한 변수
    # MAX_TRANS = 0.015
    # MAX_ROT = np.pi
    MAX_GRIP = 1100.0

    print('s: start, q: stop, t: terminate')


    while not rospy.is_shutdown():
        if terminal:
            print("Terminating.")
            break

        if latest_gripper_qpos is None:
            rospy.logwarn_throttle(5, "No gripper data")
            rate.sleep()
            continue

        if recording:
            ## qpos <- Follower
            current_jnt = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0
            qpos_joint = current_jnt[:6]  # 6자유도만 사용

            ## Δgripper 계산 (정규화) <- Follower의 gripper 값
            if len(buffer['observations']['qpos']) > 0:
                prev_grip = buffer['observations']['qpos'][-1][6] * MAX_GRIP  # 복원
                curr_grip = latest_gripper_qpos[0]
                delta_grip_norm = (curr_grip - prev_grip) / MAX_GRIP
            else:
                delta_grip_norm = 0.0

            ## qpos = joint + delta gripper <- Follower
            qpos = np.concatenate([qpos_joint, [delta_grip_norm]])
            buffer['observations']['qpos'].append(qpos)            # buffer['observations']['qvel'].append(qvel)

            # current_pose = robot.fkine(current_jnt)
            # T = np.array(current_pose)

            # robot_pos = list(T[:3, 3])
            # rot = R.from_matrix(T[:3, :3])
            # x, y, z, w = rot.as_quat()
            # robot_quat = [w, x, y, z]

            ## RGB Data
            try:
                frame0 = pipeline0.wait_for_frames(timeout_ms=2000)
                frame1 = pipeline1.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                rospy.logwarn(f"Realsense timeout: {e}")
                rate.sleep()
                continue

            color_frame0 = frame0.get_color_frame()
            color_frame1 = frame1.get_color_frame()
            if not color_frame0 or not color_frame1:
                rospy.logwarn("No color frame from one of the cameras")
                continue

            color_image0 = np.asanyarray(color_frame0.get_data())
            color_image1 = np.asanyarray(color_frame1.get_data())

            h, w = color_image0.shape[:2]
            start_x = (w - CROP_SIZE) // 2
            image0 = color_image0[:, start_x:start_x+CROP_SIZE]
            image1 = color_image1[:, start_x:start_x+CROP_SIZE]

            image0 = cv2.resize(image0, (120, 120))
            image1 = cv2.resize(image1, (120, 120))

            # for cam_name in ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']:
            # camera 여러 대 사용시 주석 해제

            buffer['observations']['images']['cam_high'].append(image0.copy())
            buffer['observations']['images']['cam_low'].append(image1.copy())


            ## action 계산 (Leader의 qpos)
            # if len(buffer['observations']['qpos']) > 1:
            #     delta_pos = (np.array(robot_pos) - np.array(buffer['observations']['qpos'][-2][:3])) / MAX_TRANS
            #     r1 = R.from_quat(robot_quat)
            #     r2 = R.from_quat(buffer['observations']['qpos'][-2][3:] if len(buffer['observations']['qpos'][-2]) == 7 else robot_quat)
            #     delta_rotvec = (r1 * r2.inv()).as_rotvec() / MAX_ROT
            #     delta_grip = (latest_gripper_qpos[0] - buffer['observations']['qpos'][-2][0]) / MAX_GRIP
            #     action = np.concatenate([delta_pos, delta_rotvec, [delta_grip]])
            # else:
            #     action = np.zeros(7)
            # action = np.concatenate([qpos, [grip]])
            action = qpos.copy()
            buffer['action'].append(action)

        elif not recording and len(buffer['action']) > 0:
            while True:
                data_store = input("Store demo data? (y/n): ").strip().lower()
                if data_store.strip() == 'y':
                    saved_i = save_to_hdf5(buffer)
                    print(f"'demo_{saved_i}' Data stored.")
                    break
                elif data_store.strip() == 'n':
                    print("Data discarded.")
                    break
                else:
                    print("Invalid input.")
            buffer = init_buffer()

        rate.sleep()

    pipeline0.stop()
    pipeline1.stop()

if __name__ == "__main__":
    main()
