import os
import time
from collections import deque

from module.config import (
    OFFSET_BTW_CENTERS,
    INFO_LEVEL,
    LINES_H_RATIO,
    MAXIMUM_ANGLUAR_SPEED_CAP,
    MAX_LINIEAR_SPEED,
    ANALOG_CAP_MODE,

    LINE_HISTORY_SIZE,
    WHITE_MODE_CONSTANT,
    YELLOW_MODE_CONSTANT,
    FOLLOW_ROAD_CROP_HALF,
)

# обработка светофора
from module.traffic_light import check_traffic_lights
# обработка поворота
from module.traffic_direction import check_direction
# обработка логов
from module.logger import log_info

import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String

import cv2
import math
import numpy as np


class Follow_Trace_Node(Node):

    def __init__(self, linear_speed=MAX_LINIEAR_SPEED):
        super().__init__("Follow_Trace_Node")

        # статус центровой точки дороги (True - актуально, False - устарела)
        self.point_status = True
        
        self._robot_Ccamera_sub = self.create_subscription(Image, "/color/image", self._callback_Ccamera, 3)
        self._pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self._robot_cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sign_finish = self.create_publisher(String ,'/robot_finish', 10)
        
        self._cv_bridge = CvBridge()
        self._msg = String()
        self.pose = Odometry()
        self.twist = Twist()

        self._linear_speed = linear_speed

        self._yellow_prevs = deque(maxlen=LINE_HISTORY_SIZE)
        self.__white_prevs = deque(maxlen=LINE_HISTORY_SIZE)
        self._yellow_prevs.append(0)
        self.__white_prevs.append(0)

        self.E = 0
        self.old_e = 0
        self.Kp = self.declare_parameter('Kp', value=3.0, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Ki = self.declare_parameter('Ki', value=1, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Kd = self.declare_parameter('Kd', value=0.25, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value

        self.STATUS_CAR = 0
        self.TASK_LEVEL = 0
        self.START_TIME = 0

        self.MAIN_LINE = "WHITE"

    # Обратный вызов для получения данных о положении
    def pose_callback(self, data):
        self.pose = data

    # Получение угла поворота из данных о положении
    def get_angle(self):
        quaternion = (self.pose.pose.pose.orientation.x, self.pose.pose.pose.orientation.y,
                      self.pose.pose.pose.orientation.z, self.pose.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        return euler[2]

    # Преобразование перспективы изображения
    def _warpPerspective(self, cvImg):
        h, w, _ = cvImg.shape
        top_x_offset = 50

        pts1 = np.float32([[0, 480], [w, 480], [top_x_offset, 300], [w-top_x_offset, 300]])
        
        # 760 x 300
        result_img_width = np.int32(abs(pts1[0][0] - pts1[1][0]))
        result_img_height = np.int32(abs(pts1[0][1] - pts1[2][0]))
        
        pts2 = np.float32([[0, 0], [result_img_width, 0], [0, result_img_height], 
                           [result_img_width, result_img_height]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(cvImg, M, (result_img_width, result_img_height))

        cv2.imshow("orig", cvImg)

        return cv2.flip(dst, 0)

    # Поиск желтой линии на изображении
    def _find_yellow_line(self, perspectiveImg_, middle_h=None):
        if FOLLOW_ROAD_CROP_HALF:
            h_, w_, _ = perspectiveImg_.shape
            perspectiveImg = perspectiveImg_[:, :w_//2, :]
        else:
            perspectiveImg = perspectiveImg_

        yellow_mask = cv2.inRange(perspectiveImg, (0, 240, 255), (0, 255, 255))
        yellow_mask = cv2.dilate(yellow_mask, np.ones((2, 2)), iterations=4)

        middle_row = yellow_mask[middle_h]
        try:
            first_notYellow = np.int32(np.where(middle_row == 255))[0][-1]
            self._yellow_prevs.append(first_notYellow)
        except:  # Используем последние данные о линии и надеемся что все починится
            first_notYellow = sum(self._yellow_prevs)//len(self._yellow_prevs)
            self.point_status = False

        return first_notYellow

    # Поиск белой линии на изображении
    def _find_white_line(self, perspectiveImg_, middle_h=None):
        fix_part = 0  # значения для исправления обрезки пополам

        if FOLLOW_ROAD_CROP_HALF:
            h_, w_, _ = perspectiveImg_.shape
            perspectiveImg = perspectiveImg_[:, w_//2:, :]
            fix_part = w_//2
        else:
            perspectiveImg = perspectiveImg_

        white_mask = cv2.inRange(
            perspectiveImg, (250, 250, 250), (255, 255, 255))

        middle_row = white_mask[middle_h]
        try:
            first_white = np.int32(np.where(middle_row == 255))[0][0]
            self.__white_prevs.append(first_white)
        except:  # Используем последние данные о линии и надеемся что все починится
            first_white = sum(self.__white_prevs)//len(self.__white_prevs)
            self.point_status = False

        return first_white + fix_part

    # Расчет новой угловой скорости с использованием PID-регулятора
    def _compute_PID(self, target):
        err = target
        e = np.arctan2(np.sin(err), np.cos(err))

        e_P = e
        e_I = self.E + e
        e_D = e - self.old_e

        w = self.Kp*e_P + self.Ki*e_I + self.Kd*e_D

        w = np.arctan2(np.sin(w), np.cos(w))

        self.E = self.E + e
        self.old_e = e
        return w

    # Обработка данных с камеры
    def _callback_Ccamera(self, msg: Image):
        self.point_status = True

        self.twist.linear.x = self._linear_speed

        # обрабатываем изображения с камеры
        cvImg = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        # получаем изображения перед колесами
        perspective = self._warpPerspective(cvImg)
        perspective_h, persective_w, _ = perspective.shape

        hLevelLine = int(perspective_h*LINES_H_RATIO)

        # получаем координаты края желтой линии и белой
        if self.MAIN_LINE == "WHITE":
            endYellow = WHITE_MODE_CONSTANT
        else:
            endYellow = self._find_yellow_line(perspective, hLevelLine)
        
        if self.MAIN_LINE == "YELLOW":
            startWhite = YELLOW_MODE_CONSTANT
        else:
            startWhite = self._find_white_line(perspective, hLevelLine)

        middle_btw_lines = (startWhite + endYellow) // 2

        center_crds = (persective_w//2, hLevelLine)
        lines_center_crds = (middle_btw_lines, hLevelLine)

        # центр справа - положительно, центр слева - отрицательно
        direction = center_crds[0] - lines_center_crds[0]

        # обработка первой таски, светофор
        if self.TASK_LEVEL == 0:
            check_traffic_lights(self, cvImg)

        # обработка перекрестка
        if self.TASK_LEVEL == 1:
            self.START_TIME = time.time()
            check_direction(self, cvImg)
        
        # попытка остановиться
        if self.TASK_LEVEL == 2 and (time.time() - self.START_TIME) > 15:
            # self.STATUS_CAR = 0 - изменять необязательно, можно ботика не останавливать
            self._msg.data = "autorace_data"
            self._sign_finish.publish(self._msg)   
        
        # Выравниваем ботика если центры расходятся больше чем нужно
        if (abs(direction) > OFFSET_BTW_CENTERS):
            angle_to_goal = math.atan2(direction, 215)
            angular_v = self._compute_PID(angle_to_goal)
            self.twist.angular.z = angular_v

            if ANALOG_CAP_MODE:
                angular_v *= 3/4
            self.twist.linear.x = abs(self._linear_speed * (MAXIMUM_ANGLUAR_SPEED_CAP - abs(angular_v)))
        

        if INFO_LEVEL:
            persective_drawed = cv2.rectangle(
                perspective, center_crds, center_crds, (0, 255, 0), thickness=10)  # Центр изображения
            if self.point_status:
                persective_drawed = cv2.rectangle(
                    persective_drawed, lines_center_crds, lines_center_crds, (0, 0, 255), thickness=10)  # центр точки между линиями
            else:
                persective_drawed = cv2.rectangle(
                    persective_drawed, lines_center_crds, lines_center_crds, (50, 50, 50), thickness=10)  # центр точки между линиями
            
            # Пытаемся соединить центр изображения с центром между линиями
            cv2.imshow("img", persective_drawed)
            cv2.waitKey(1)


        if self.STATUS_CAR == 1:
            self._robot_cmd_vel_pub.publish(self.twist)


def main():
    rclpy.init()
    FTN = Follow_Trace_Node()
    rclpy.spin(FTN)
    FTN.destroy_node()
    rclpy.shutdown()