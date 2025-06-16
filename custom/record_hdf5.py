#!/usr/bin/env python3
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

f = h5py.File('/home/vision/catkin_ws/src/teleop_data/data/tele_data_0611.hdf5', 'w')
data = f.create_group('data')

def init_buffer():
    return {
        'demo_0': {'actions': []},
        'obs': {
            'robot_eef_pos': [],
            'robot_eef_quat': [],
            'robot_gripper_qpos': [],
            'image': []
        }
    }

def make_demo_n(buffer):
    demo_n = data.create_group(f'demo_{len(data.keys())}')
    obs = demo_n.create_group('obs')

    for name, values in buffer['demo_0'].items():
        demo_n.create_dataset(name, data=np.array(values))
    for name, values in buffer['obs'].items():
        obs.create_dataset(name, data=np.array(values))

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

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

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
    rate = rospy.Rate(20)

    MAX_TRANS = 0.015
    MAX_ROT = np.pi
    MAX_GRIP = 1100.0

    while not rospy.is_shutdown():
        if terminal:
            f.close()
            print("Terminating.")
            break

        if latest_gripper_qpos is None:
            rospy.logwarn_throttle(5, "No gripper data")
            rate.sleep()
            continue

        if recording:
            # qpos
            current_jnt = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0
            current_pose = robot.fkine(current_jnt)
            T = np.array(current_pose)

            robot_pos = list(T[:3, 3])
            rot = R.from_matrix(T[:3, :3])
            x, y, z, w = rot.as_quat()
            robot_quat = [w, x, y, z]

            # RGB Data
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                rospy.logwarn(f"Realsense timeout: {e}")
                rate.sleep()
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                rospy.logwarn("No color frame")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            start_x = (w - CROP_SIZE) // 2
            image = color_image[:, start_x:start_x+CROP_SIZE]

            # delta 값 계산
            # delta_pos
            if len(buffer['obs']['robot_eef_pos']) > 0:
                prev_pos = np.array(buffer['obs']['robot_eef_pos'][-1])
                delta_pos = (np.array(robot_pos) - prev_pos) / MAX_TRANS
            else:
                delta_pos = np.zeros(3)
            
            # delta_quat
            if len(buffer['obs']['robot_eef_quat']) > 0:
                prev_quat = buffer['obs']['robot_eef_quat'][-1]
                r1 = R.from_quat([prev_quat[1], prev_quat[2], prev_quat[3], prev_quat[0]])
                r2 = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
                delta_rotvec = (r2 * r1.inv()).as_rotvec() / MAX_ROT
            else:
                delta_rotvec = np.zeros(3)
            
            # gripper
            if len(buffer['obs']['robot_gripper_qpos']) > 0:
                prev_grip = buffer['obs']['robot_gripper_qpos'][-1][0]
                curr_grip = latest_gripper_qpos[0]
                delta_grip = (curr_grip - prev_grip) / MAX_GRIP
            else:
                delta_grip = 0.0
            
            # action
            action = np.concatenate([delta_pos, delta_rotvec, [delta_grip]])

            # demonstration data
            buffer['demo_0']['actions'].append(action)
            buffer['obs']['robot_eef_pos'].append(robot_pos)
            buffer['obs']['robot_eef_quat'].append(robot_quat)
            buffer['obs']['robot_gripper_qpos'].append(latest_gripper_qpos)
            buffer['obs']['image'].append(image.copy())

        elif not recording and len(buffer['demo_0']['actions']) > 0:
            while True:
                data_store = input("Store demo data? (y/n): ").strip().lower()
                if data_store == 'y':
                    make_demo_n(buffer)
                    print("Data stored.")
                    break
                elif data_store == 'n':
                    print("Data discarded.")
                    break
                else:
                    print("Invalid input.")

            buffer = init_buffer()

        rate.sleep()

    pipeline.stop()

if __name__ == "__main__":
    main()