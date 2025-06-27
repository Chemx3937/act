#  send_joint_command()와 send_gripper_command() 함수 구현 (실제 API 또는 통신 방식)

#  초기화 및 리셋 함수 (move_arms, move_grippers)도 환경에 맞게 구성

#  inference에서 get_qpos() → policy → set_joint_positions, set_gripper_pose()로 연결되는 흐름 구성

import pyrealsense2 as rs
import numpy as np

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


class Recorder:
    def __init__(self):
        self.latest_joint = None
        self.latest_gripper_qpos = None

    def update_joint(self):
        # 사용자 시스템에서 실시간 조인트 값 받아오는 함수
        self.latest_joint = np.array(GetCurrentSplitedJoint()) * np.pi / 180.0

    def update_gripper(self, gripper_msg):
        self.latest_gripper_qpos = gripper_msg.rGWD

    def get_qpos(self):
        self.update_joint()
        if self.latest_gripper_qpos is None:
            raise RuntimeError("그리퍼 데이터가 아직 없음")
        qpos_joint = self.latest_joint[:6]
        norm_grip = (self.latest_gripper_qpos - PREV_GRIP) / MAX_GRIP
        return np.concatenate([qpos_joint, [norm_grip]])
    
    def set_joint_positions(self, target_joint):
        # 사용자 로봇 SDK나 API로 target_joint (길이 6) 값을 전송
        send_joint_command(target_joint)

    def set_gripper_pose(self, norm_grip_val):
        # 정규화된 값을 원래 값으로 역변환
        grip_command = norm_grip_val * MAX_GRIP + MIN_GRIP
        send_gripper_command(grip_command)
