#  send_joint_command()와 send_gripper_command() 함수 구현 (실제 API 또는 통신 방식)

#  초기화 및 리셋 함수 (move_arms, move_grippers)도 환경에 맞게 구성

#  inference에서 get_qpos() → policy → set_joint_positions, set_gripper_pose()로 연결되는 흐름 구성

import pyrealsense2 as rs
import numpy as np


def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    if len(serials) < 2:
        raise RuntimeError("2개 이상의 Realsense 카메라가 연결되어 있어야 합니다.")
    print("Detected serials:", serials)
    return serials

serials = get_device_serials()
serial_d435 = serials[0]
serial_d405 = serials[1]

class ImageRecorder:
    def __init__(self, serial_d435, serial_d405):
        self.pipeline0 = rs.pipeline()
        config0 = rs.config()
        config0.enable_device(serial_d435)
        config0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline0.start(config0)

        self.pipeline1 = rs.pipeline()
        config1 = rs.config()
        config1.enable_device(serial_d405)
        config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline1.start(config1)

    def get_images(self):
        frames0 = self.pipeline0.wait_for_frames(timeout_ms=2000)
        frames1 = self.pipeline1.wait_for_frames(timeout_ms=2000)
        color0 = frames0.get_color_frame()
        color1 = frames1.get_color_frame()

        image0 = np.asanyarray(color0.get_data()) if color0 else None
        image1 = np.asanyarray(color1.get_data()) if color1 else None

        return {'cam_high': image0, 'cam_low': image1}


import numpy as np
import rospy
import time

# 둘중 어떤 거 사용하는지 확인 필요
# from teleop_data.msg import OnRobotRGOutput
from onrobot_rg_control.msg import OnRobotRGOutput


import sys
sys.path.append('/home/vision/catkin_ws/src/robotory_rb10_rt/scripts')  # 필요시 조정
from api.cobot import GetCurrentSplitedJoint, SendCOMMAND, CMD_TYPE

MAX_GRIP = 1100.0


class Recorder:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.qpos = None
        self.joint_deg = None
        self.curr_gripper = None       # rGWD: 현재 상태
        self.prev_gripper = None       # 이전 상태
        self.gripper_pub = None

        if init_node:
            rospy.init_node("custom_recorder", anonymous=True)
        rospy.Subscriber("/OnRobotRGOutput", OnRobotRGOutput, self._gripper_cb)

        if self.is_debug:
            from collections import deque
            self.gripper_log = deque(maxlen=50)
        time.sleep(0.1)

    def _gripper_cb(self, msg):
        self.curr_gripper = msg.rGWD
        if self.prev_gripper is None:
            self.prev_gripper = self.curr_gripper
        if self.is_debug:
            self.gripper_log.append(time.time())

    def update_joint(self):
        self.joint_deg = GetCurrentSplitedJoint()

    def get_qpos(self):
        """
        조인트 각도 (rad) + Δgripper 정규화된 값 반환
        """
        self.update_joint()
        joint_rad = np.array(self.joint_deg[:6]) * np.pi / 180.0

        if self.curr_gripper is None:
            raise RuntimeError("Gripper data not received yet")

        # MAX_GRIP = 1100.0
        if self.prev_gripper is None:
            delta_grip_norm = 0.0
        else:
            delta_grip_norm = (self.curr_gripper - self.prev_gripper) / MAX_GRIP
        self.prev_gripper = self.curr_gripper

        return np.concatenate([joint_rad, [delta_grip_norm]])

    def set_joint_positions(self, joint_rad: np.ndarray):
        """
        joint_rad (6,) → degree 변환 → cobot API 전송
        """
        if joint_rad.shape != (6,):
            raise ValueError("Expected shape (6,), got", joint_rad.shape)

        joint_deg = joint_rad * 180 / np.pi
        msg = f"move_servo_j(jnt[{','.join(f'{j:.3f}' for j in joint_deg)}],0.002,0.1,0.02,0.2)"
        # ServoJ(joint_deg, time1=0.002, time2=0.1, gain=0.02, lpf_gain=0.2)
        SendCOMMAND(msg, CMD_TYPE.MOVE)

    def set_gripper_pose(self, delta_grip_norm: float):
        """
        Δgripper 정규화 값을 받아 이전 상태 기준으로 이동
        """

        if self.curr_gripper is None:
            raise RuntimeError("Gripper state not initialized")

        # 목표 gripper width 계산
        target_gripper = self.curr_gripper + delta_grip_norm * MAX_GRIP
        target_gripper = np.clip(target_gripper, 0, MAX_GRIP)

        # publisher가 없으면 생성
        if self.gripper_pub is None:
            self.gripper_pub = rospy.Publisher("OnRobotRGOutput", OnRobotRGOutput, queue_size=10)
            time.sleep(0.1)

        # 명령 메시지 생성
        cmd = OnRobotRGOutput()
        cmd.rGWD = int(target_gripper)  # 목표 폭
        cmd.rGFR = 400                 # 일정한 그립 force
        cmd.rCTR = 16                  # position control mode

        # publish
        self.gripper_pub.publish(cmd)

        # 상태 갱신
        self.prev_gripper = self.curr_gripper
        self.curr_gripper = target_gripper

    def print_diagnostics(self):
        if self.is_debug and len(self.gripper_log) > 1:
            diffs = np.diff(np.array(self.gripper_log))
            print(f"[Gripper ROS Hz] ~{1 / np.mean(diffs):.2f} Hz")
        else:
            print("No gripper diagnostics available.")

    # 추가해야 되는 것
    # reset 시킬때만 Joint 움직이게 하는 메소드
    def move_arm(self, target_pose, move_time=1.0):
        from custom_constants import DT

        """
        로봇 조인트를 target_pose로 천천히 이동 (reset용)
        :param target_pose: (6,) array, radian 단위
        """
        num_steps = int(move_time / DT)
        current_pose = self.get_qpos()[:6]
        traj = np.linspace(current_pose, target_pose, num_steps)
        for step_pose in traj:
            self.set_joint_positions(step_pose)
            time.sleep(DT)

    def move_gripper(self, target_grip, move_time=1.0):
        from custom_constants import DT
        """
        gripper를 target_grip (절대 width 값, rGWD 기준)으로 천천히 이동 (reset용)
        :param target_grip: int or float (절대 단위, 0 ~ MAX_GRIP)
        """
        num_steps = int(move_time / DT)
        if self.curr_gripper is None:
            rospy.logwarn("Gripper state not initialized; skipping move_gripper")
            return

        curr_grip = self.curr_gripper
        traj = np.linspace(curr_grip, target_grip, num_steps)
        for g in traj:
            delta_grip_norm = (g - self.curr_gripper) / MAX_GRIP
            self.set_gripper_pose(delta_grip_norm)
            time.sleep(DT)
