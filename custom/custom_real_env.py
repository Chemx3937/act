# 1. self.follower_bot 정의 <- rb, gripper와 연결하는 클래스로 custom_robot_utils.py에서 정의
# 2. Dual Arm -> Single Arm으로 변경 (완)
# 3. 제어 명령어들 수정
# 4. 사용하지 않는 observation들 처리 (완)
# 5. RB10, 그리퍼 Data Subscribe <- get_
# 6. RB10, 그리퍼에 Action Publish <- step 메소드에서 실행


import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from custom_constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from custom_constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from custom_constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
# from custom_robot_utils import Recorder, ImageRecorder
from test_custom_robot_utils import Recorder, ImageRecorder, MAX_GRIP

from custom_robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
# from interbotix_xs_modules.recorder_arm import InterbotixManipulatorXS
# from interbotix_xs_msgs.msg import JointSingleCommand

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot single manipulation
    Action space:      [qpos (6),             # absolute joint position
                        gripper_positions (1),    # normalized gripper position (0: close, 1: open)]

    Observation space: {"qpos": Concat[ qpos (6),          # absolute joint position
                                        gripper_position (1)],  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ zeros]  #사용 X
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3)},         # h, w, c, dtype='uint8'
    """

    # def __init__(self, init_node, setup_robots=True):
    def __init__(self, init_node):
        # self.follower_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="recorder_arm", gripper_name="gripper",
        #                                                robot_name=f'puppet_left', init_node=init_node)

        # self,recorder_arm 부분에서 Recoder 클래스로 rb, 그리퍼 연결하므로 필요 X
        # if setup_robots:
        #     self.setup_robots()

        # RB,Gripper의 state, 제어 명령을 내릴수 있게 연결시키는 Recorder 클래스 객체 생성
        # self.recorder_arm = Recorder(init_node=False)
        self.recorder_arm = Recorder(init_node=init_node)

        # self.recorder_arm = Recorder('left', init_node=False)
        # self.recorder_right = Recorder('right', init_node=False)
        # realsense 연결시키는 ImageRecorder 클래스 객체 생성
        # self.image_recorder = ImageRecorder(init_node=False)
        self.image_recorder = ImageRecorder(init_node=init_node)

        # JointSingleCommand는 interbotix_xs_msgs.msg로 aloha에서만 사능한 명령어이므로 사용 X
        # self.gripper_command = JointSingleCommand(name="gripper") 

    # def setup_robots(self):
    #     setup_puppet_bot(self.follower_bot)

    # 원본
    # def get_qpos(self):
    #     qpos_raw = self.recorder_arm.qpos
    #     follower_arm_qpos = qpos_raw[:6]
    #     follower_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[7])] # this is position not joint
    #     return np.concatenate([follower_arm_qpos, follower_gripper_qpos])

    # 수정 <- custom_robot_utils.py에 있는 get_qpos 사용하도록 변경
    def get_qpos(self):
        return self.recorder_arm.get_qpos()

    # def get_qvel(self):
    #     qvel_raw = self.recorder_arm.qvel
    #     follower_arm_qvel = qvel_raw[:6]
    #     follower_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[7])]
    #     return np.concatenate([follower_arm_qvel, follower_gripper_qvel])

    # def get_effort(self):
    #     effort_raw = self.recorder_arm.effort
    #     follower_robot_effort = effort_raw[:7]
    #     return np.concatenate([follower_robot_effort])

    def get_images(self):
        return self.image_recorder.get_images()

    # 원본
    # def set_gripper_pose(self, desired_pos_normalized):
    #     gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(desired_pos_normalized)
    #     self.gripper_command.cmd = gripper_desired_joint
    #     self.follower_bot.gripper.core.pub_single.publish(self.gripper_command)

    # 수정 <- robot_utils.py에서 정의

    # custom_constatns에서 정의한 START_ARM_POSE로 움직이게 함
    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        self.recorder_arm.move_arm(reset_position, move_time=1.0)
        # move_arms([self.follower_bot], [reset_position], move_time=1)

    # 그리퍼가 완전히 열리게 함
    def _reset_gripper(self):
        """
        1. 절반 닫기 (폭 550)
        2. 완전히 열기 (폭 1100)
        """
        half_closed = 550.0
        fully_open = 1100.0

        self.recorder_arm.move_gripper(half_closed, move_time=0.5)
        self.recorder_arm.move_gripper(fully_open, move_time=1.0)


    # def _reset_gripper(self):
    #     """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
    #     move_grippers([self.follower_bot], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
    #     move_grippers([self.follower_bot], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        # obs['qvel'] = self.get_qvel()
        # obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            # self.follower_bot.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    # custom_robot_utils.py에서 새로 정의한 
    # step별 action을 실행시키는 부분으로 로봇 움직이게 함
    def step(self, action):
        # state_len = int(len(action) / 2)
        # follower_action = action[:state_len]
        # # left_action = action[:state_len]
        # # right_action = action[state_len:]

        follower_action = action

        # rb, 그리퍼에 action 적용시켜서 움직이게 하는 부분
        # self.follower_bot.recorder_arm.set_joint_positions(left_action[:6], blocking=False)
        # self.follower_bot.recorder_arm.set_joint_positions(follower_action[:6], blocking=False)
        self.recorder_arm.set_joint_positions(follower_action[:6])
        self.recorder_arm.set_gripper_pose(follower_action[-1])
        # self.set_gripper_pose(left_action[-1], right_action[-1])
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

# custom에서 사용 X
def get_action(Leader_bot):
    action = np.zeros(14) # 6 joint + 1 gripper, for two recorder_arms
    # recorder_arm actions
    action[:6] = Leader_bot.dxl.joint_states.position[:6]
    # action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(Leader_bot.dxl.joint_states.position[6])
    # action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


# def make_real_env(init_node, setup_robots=True):
#     env = RealEnv(init_node, setup_robots)
#     return env

def make_real_env(init_node):
    env = RealEnv(init_node)
    return env


# custom에서 사용 X
def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master recorder_arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # source of data
    Leader_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="recorder_arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    # master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="recorder_arm", gripper_name="gripper",
    #                                            robot_name=f'master_right', init_node=False)
    setup_master_bot(Leader_bot)
    # setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        # action = get_action(Leader_bot, master_bot_right)
        action = get_action(Leader_bot)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    # test_real_teleop()
    pass

