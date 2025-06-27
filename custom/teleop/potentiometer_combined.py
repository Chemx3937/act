#!/usr/bin/env python3
import rospy
import serial
from std_msgs.msg import Int32
from onrobot_rg_control.msg import OnRobotRGOutput
from onrobot_rg_control.msg import OnRobotRGInput

class PotentiometerGripperController:
    def __init__(self):
        rospy.init_node('potentiometer_gripper_controller', anonymous=True)
        rospy.loginfo("Starting potentiometer gripper controller node...")

        # Serial 설정
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.last_value = -1
        self.rate = rospy.Rate(20)

        # 그리퍼 타입 파라미터
        self.gtype = rospy.get_param('/onrobot/gripper', 'rg2')
        # if self.gtype == 'rg2':
        self.max_force = 400
        self.max_width = 1100
        # else:
        #     self.max_force = 1200
        #     self.max_width = 1600

        # 캘리브레이션 파라미터
        self.pot_min = rospy.get_param('~pot_min', 1)
        self.pot_max = rospy.get_param('~pot_max', 167)

        # 퍼블리셔
        self.pub_gripper = rospy.Publisher('OnRobotRGOutput', OnRobotRGOutput, queue_size=10)
        # rospy.Subscriber('OnRobotRGInput',OnRobotRGInput,gripper.refreshCommand, queue_size=1)

        # 그리퍼 커맨드 초기화
        self.cmd = OnRobotRGOutput()
        self.cmd.rGFR = self.max_force
        self.cmd.rCTR = 16

        # status = self.gripper.getStatus()

    def read_and_control(self):
        while not rospy.is_shutdown():
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode().strip()
                    # print(line)
                    pot_value = int(''.join(filter(str.isdigit, line)))
                    
                    # 노이즈 필터링
                    if abs(pot_value - self.last_value) > -1:
                        self.last_value = pot_value

                        # pot → clamped → norm → width 변환
                        pot_clamped = max(self.pot_min, min(self.pot_max, pot_value))
                        norm = float(pot_clamped - self.pot_min) / (self.pot_max - self.pot_min)
                        width = int(norm * self.max_width)

                        # force_th = abs(pot_clamped - status.gGWD)

                        # 그리퍼 명령 설정 및 퍼블리시
                        self.cmd.rGWD = width
                        # self.cmd.rGFR = width - 400/force_th
                        self.pub_gripper.publish(self.cmd)

                        rospy.loginfo(f'Analog: {pot_value} → Clamped: {pot_clamped}, Norm: {norm:.3f} → Width: {width}')

                except ValueError:
                    rospy.logwarn(f'Invalid serial input: {line}')
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = PotentiometerGripperController()
        controller.read_and_control()
    except rospy.ROSInterruptException:
        pass
