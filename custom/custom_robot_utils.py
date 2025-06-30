# 1. ImageRecorder 클래스에서 ros로 이미지 불러오는 게 아닌 realsense에서 바로 받아오게 변경
# 2. rb, gripper와 연결하는 클래스 정의
# 3. Recorder 클래스 rb와 그리퍼의 state 읽을 수 있도록 정의
# 4. rb, 그리퍼 제어하는 명령어 정의 <- set_joint_positions, set_gripper_pose 라는 메소드로 rb, 그리퍼 움직이게 하기


import numpy as np
import time
from constants import DT
from interbotix_xs_msgs.msg import JointSingleCommand

import pyrealsense2 as rs


import IPython
e = IPython.embed

# 1. ImageRecorder 클래스에서 ros로 이미지 불러오는 게 아닌 realsense에서 바로 받아오게 변경 필요
class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        # self.camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        self.camera_names = ['cam_high', 'cam_low',]


        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/usb_{cam_name}/image_raw", Image, callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        # cv2.imwrite('/home/tonyzhao/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def image_cb_cam_high(self, data):
        cam_name = 'cam_high'
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = 'cam_low'
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = 'cam_left_wrist'
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = 'cam_right_wrist'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()

# 2. rb, gripper와 연결하는 클래스 정의 필요
# rb,그리퍼 state 읽을수 있게 정의, 제어 명령 보낼 수 있게 정의

# 3. Recorder 클래스 rb와 그리퍼의 state 읽을 수 있도록 정의 필요 <- rb: 자체 명령어, 그리퍼: ROS
# rb: GetCurrentSplitedJoint
# 그리퍼: rospy.Subscriber("/OnRobotRGOutput", OnRobotRGOutput, gripper_callback)

# 4. rb, 그리퍼 제어하는 명령어 정의 필요


class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        # self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug
        
        # 추가 이전 step 그리퍼 state 저장
        self.prev_gripper_position = None


        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_group", JointGroupCommand, self.puppet_arm_commands_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_single", JointSingleCommand, self.puppet_gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    # Manipulator, Gripper Joint State 읽어오는 callback 함수
    def puppet_state_cb(self, data):
        self.qpos = data.position
        # self.qvel = data.velocity
        # self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    # Manipulator 움직이게 하는 명령어를 읽는 callback 함수
    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    # 그리퍼 움직이게 하는 명령어를 읽는 callback 함수
    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    # Manipulator와 Gripper의 현재 state와 이들을 목표로 움직이게 하는 명령어를 비교해 실시간 제어가 되는 지 확인해주는 메소드
    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')

    # 새로 추가
    # Joint 움직이게 하는 메소드
    def set_joint_positions():
        return

    # 새로 추가
    # 그리퍼 움직이게 하는 메소드
    def set_gripper_pose(self, desired_pos_normalized):
        return 

# 수정 필요
def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

# 수정 필요
def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position



# reset 시킬때만 Joint 움직이게 하는 메소드
# 수정 필요
def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            # bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
            bot.set_joint_positions(traj_list[bot_id][t])
        time.sleep(DT)

# reset 시킬때만 그리퍼 움직이게 하는 메소드
# 수정 필요
def move_grippers(bot_list, target_pose_list, move_time):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

def setup_puppet_bot(bot):
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)
