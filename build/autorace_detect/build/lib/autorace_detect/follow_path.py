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
    FOLLOW_ROAD_MODE,
)

# обработка светофора
from module.traffic_light import check_traffic_lights
# обработка поворота
from module.traffic_direction import check_direction, analyze_arrow_direction
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

        self._yellow_right_edge = deque(maxlen=LINE_HISTORY_SIZE)  # Правый край желтой линии
        self._white_left_edge = deque(maxlen=LINE_HISTORY_SIZE)    # Левый край белой линии
        
        # Инициализируем с разумными значениями
        self._yellow_right_edge.append(100)   # Желтая начинается примерно на 100 пикселе
        self._white_left_edge.append(500)     # Белая начинается примерно на 500 пикселе

        self.E = 0
        self.old_e = 0
        self.Kp = self.declare_parameter('Kp', value=1.5, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Ki = self.declare_parameter('Ki', value=0.05, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Kd = self.declare_parameter('Kd', value=0.3, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value

        self.STATUS_CAR = 0
        self.TASK_LEVEL = 0
        self.START_TIME = 0

        self.MAIN_LINE = FOLLOW_ROAD_MODE
        self.image_width = 0
        self.image_height = 0

        # ДОБАВЛЯЕМ: состояние остановки
        self._is_stopped = False
        self._stop_start_time = 0
        self._stop_duration = 3.0  # секунды остановки
        self._stop_reason = ""  # причина остановки
        
        # ДОБАВЛЯЕМ: флаги для управления перекрестком
        self._stop_sign_detected = False
        self._direction_determined = False
        self._ignore_stop_sign = False  # Игнорировать знак STOP после первой остановки
        
        # ДОБАВЛЯЕМ: время последней проверки
        self._last_check_time = 0
        self._check_interval = 0.2  # проверять каждые 0.2 секунды

    def _stop_robot(self, reason="STOP_SIGN"):
        log_info(self, f"[СТОП] Останавливаю робота: {reason}")
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self._robot_cmd_vel_pub.publish(self.twist)
        self._is_stopped = True
        self._stop_start_time = time.time()
        self._stop_reason = reason
        
        # Если это знак STOP, устанавливаем флаги
        if reason == "STOP_SIGN":
            self._stop_sign_detected = True
        
        # Визуализация остановки
        if INFO_LEVEL:
            stop_img = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(stop_img, f"STOPPED: {reason}", (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(stop_img, f"Waiting {self._stop_duration}s", (60, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Stop Status", stop_img)
            cv2.waitKey(1)

    # Функция возобновления движения
    def _resume_robot(self):
        log_info(self, "[СТОП] Возобновляю движение")
        self.twist.linear.x = self._linear_speed
        self._robot_cmd_vel_pub.publish(self.twist)
        self._is_stopped = False
        
        if INFO_LEVEL:
            cv2.destroyWindow("Stop Status")

    # Обработка остановки
    def _handle_stop(self):
        if not self._is_stopped:
            return False  # Робот не остановлен
        
        current_time = time.time()
        elapsed = current_time - self._stop_start_time
        
        # Обновляем таймер
        if INFO_LEVEL and elapsed < self._stop_duration:
            stop_img = np.zeros((200, 400, 3), dtype=np.uint8)
            remaining = self._stop_duration - elapsed
            cv2.putText(stop_img, f"STOPPED: {self._stop_reason}", (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(stop_img, f"Resume in: {remaining:.1f}s", (60, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Stop Status", stop_img)
            cv2.waitKey(1)
        
        # Проверяем, прошло ли достаточно времени
        if elapsed >= self._stop_duration:
            # После остановки на знаке STOP, игнорируем его в будущем
            if self._stop_reason == "STOP_SIGN":
                self._ignore_stop_sign = True
                log_info(self, "[СТОП] Игнорирую знак STOP до конца перекрестка")
            
            self._resume_robot()
            return False  # Остановка завершена
        
        return True  # Робот все еще остановлен

    # Проверка интервала проверки
    def _should_check(self):
        current_time = time.time()
        if current_time - self._last_check_time < self._check_interval:
            return False
        self._last_check_time = current_time
        return True

    # Определение направления ВО ВРЕМЯ остановки
    def _determine_direction_during_stop(self, img):
        """Определяет направление поворота пока робот остановлен"""
        if not self._is_stopped or self._direction_determined:
            return
        
        # Анализируем направление знака
        direction = analyze_arrow_direction(img)  # ИЗМЕНИТЬ ТУТ!
        
        if direction == "LEFT":
            log_info(self, "[СТОП] Определил направление: ПОВОРОТ НАЛЕВО")
            self.MAIN_LINE = "YELLOW"
            self._direction_determined = True
        elif direction == "RIGHT":
            log_info(self, "[СТОП] Определил направление: ПОВОРОТ НАПРАВО")
            self.MAIN_LINE = "WHITE"
            self._direction_determined = True
        elif direction == "STRAIGHT":
            log_info(self, "[СТОП] Определил направление: ПРЯМО")
            self.MAIN_LINE = "WHITE"
            self._direction_determined = True
        else:
            log_info(self, "[СТОП] Направление не определено, буду поворачивать налево")
            self.MAIN_LINE = "YELLOW"
            self._direction_determined = True

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
        self.image_width = w
        self.image_height = h
        
        # Точки для перспективного преобразования
        top_x_offset = 80  # Увеличил для лучшего обзора
        
        # Берем нижнюю часть изображения и преобразуем в вид сверху
        pts1 = np.float32([
            [0, h-1],           # Левый нижний
            [w-1, h-1],         # Правый нижний
            [top_x_offset, 350], # Левый верхний (поднял выше)
            [w-top_x_offset, 350]  # Правый верхний (поднял выше)
        ])
        
        # Вычисляем размеры выходного изображения
        result_img_width = 640  # Фиксированная ширина
        result_img_height = 480  # Фиксированная высота
        
        pts2 = np.float32([
            [0, result_img_height-1],
            [result_img_width-1, result_img_height-1],
            [0, 0],
            [result_img_width-1, 0]
        ])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(cvImg, M, (result_img_width, result_img_height))

        return dst

    # Поиск желтой линии на изображении
    def _find_yellow_line(self, perspectiveImg, middle_h=None):
        h, w, _ = perspectiveImg.shape
        
        if middle_h is None:
            middle_h = h // 2
        
        # Ищем желтый цвет в нижней части изображения (более стабильно)
        search_height = middle_h
        search_region = perspectiveImg[search_height-20:search_height+20, :]
        
        # Конвертируем в HSV для лучшего определения желтого
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Диапазон для желтого цвета
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Морфологические операции для очистки
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Находим правый край контура (ближайший к центру дороги)
            rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            
            # Корректируем координату по высоте
            yellow_edge = rightmost_point[0]
            
            # Фильтруем выбросы
            if len(self._yellow_right_edge) > 0:
                last_value = self._yellow_right_edge[-1]
                if abs(yellow_edge - last_value) < 100:  # Не допускаем резких скачков
                    self._yellow_right_edge.append(yellow_edge)
                    self.point_status = True
                    return yellow_edge
            
            # Если значение странное, используем последнее хорошее
            if len(self._yellow_right_edge) > 0:
                return self._yellow_right_edge[-1]
        
        # Если не нашли желтую линию
        if len(self._yellow_right_edge) > 0:
            self.point_status = False
            return self._yellow_right_edge[-1]  # Используем последнее известное значение
        else:
            return 150  # Значение по умолчанию

    # Поиск белой линии на изображении
    def _find_white_line(self, perspectiveImg, middle_h=None):
        h, w, _ = perspectiveImg.shape
        
        if middle_h is None:
            middle_h = h // 2
        
        # Ищем белую линию в нижней части
        search_height = middle_h
        search_region = perspectiveImg[search_height-20:search_height+20, :]
        
        # Конвертируем в серый
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Адаптивный порог для лучшего выделения белого
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Также используем HSV для белого
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        white_mask_hsv = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        mask = cv2.bitwise_or(mask, white_mask_hsv)
        
        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Берем самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Находим левый край контура (ближайший к центру дороги)
            leftmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            
            # Корректируем координату по высоте
            white_edge = leftmost_point[0]
            
            # Фильтруем выбросы
            if len(self._white_left_edge) > 0:
                last_value = self._white_left_edge[-1]
                if abs(white_edge - last_value) < 100:  # Не допускаем резких скачков
                    self._white_left_edge.append(white_edge)
                    self.point_status = True
                    return white_edge
            
            # Если значение странное, используем последнее хорошее
            if len(self._white_left_edge) > 0:
                return self._white_left_edge[-1]
        
        # Если не нашли белую линию
        if len(self._white_left_edge) > 0:
            self.point_status = False
            return self._white_left_edge[-1]  # Используем последнее известное значение
        else:
            return 500  # Значение по умолчанию

    # Расчет новой угловой скорости с использованием PID-регулятора
    def _compute_PID(self, error):
        # Нормализуем ошибку
        e = np.arctan2(np.sin(error), np.cos(error))

        e_P = e
        e_I = self.E + e
        e_D = e - self.old_e

        w = self.Kp * e_P + self.Ki * e_I + self.Kd * e_D

        # Ограничиваем выход
        w = max(min(w, 1.5), -1.5)
        
        # Обновляем интегральную составляющую
        self.E = self.E + e
        self.old_e = e
        
        return w

    # Обработка данных с камеры (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    def _callback_Ccamera(self, msg: Image):
        # 1. Проверяем, не остановлен ли робот
        if self._handle_stop():
            # Если робот остановлен, все равно получаем изображение для анализа направления
            cvImg = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
            
            # Определяем направление пока остановлен
            if self.TASK_LEVEL == 1 and not self._direction_determined:
                self._determine_direction_during_stop(cvImg)
            
            return
        
        # 2. Проверяем интервал проверки
        if not self._should_check():
            return
        
        # 3. Получаем изображение
        cvImg = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        # 4. Проверяем угол для определения положения
        angle = self.get_angle()
        is_at_intersection = abs(angle) >= 2.80
        
        # 5. Обработка светофора
        if self.TASK_LEVEL == 0:
            check_traffic_lights(self, cvImg)

        # 6. Устанавливаем скорость движения
        self.twist.linear.x = self._linear_speed

        # 7. Получаем изображения перед колесами
        perspective = self._warpPerspective(cvImg)
        perspective_h, persective_w, _ = perspective.shape

        hLevelLine = int(perspective_h * LINES_H_RATIO)

        # 8. Находим линии
        yellow_edge = self._find_yellow_line(perspective, hLevelLine)
        white_edge = self._find_white_line(perspective, hLevelLine)
        
        # 9. Проверяем корректность линий
        if white_edge <= yellow_edge + 50:
            if self.point_status:
                white_edge = yellow_edge + 200
            else:
                white_edge = min(persective_w - 50, yellow_edge + 200)

        # 10. Вычисляем центр дороги и ошибку
        road_center = (yellow_edge + white_edge) / 2
        image_center = persective_w / 2
        error = image_center - road_center
        normalized_error = error / (persective_w / 2)

        # 11. Обработка перекрестка (TASK_LEVEL == 1)
        if self.TASK_LEVEL == 1:
            if self.START_TIME == 0:
                self.START_TIME = time.time()
                log_info(self, f"[ПЕРЕКРЕСТОК] Начинаем анализ перекрестка")
            
            # Если робот на перекрестке, проверяем дорожные знаки
            if is_at_intersection:
                # Проверяем дорожные знаки, игнорируем STOP если уже останавливались
                result = check_direction(self, cvImg, 
                                        is_at_intersection=True,
                                        ignore_stop_sign=self._ignore_stop_sign)
                
                if result == "STOP_SIGN":
                    # Обнаружен знак остановки - останавливаем робота
                    self._stop_robot(reason="STOP_SIGN")
                    return
                elif result == "DIRECTION_DECIDED":
                    # Направление определено, переходим к следующему этапу
                    log_info(self, f"[ПЕРЕКРЕСТОК] Направление определено: {self.MAIN_LINE}")
                    return
                elif not self._direction_determined:
                    # Если направление еще не определено, пытаемся его определить
                    self._determine_direction_during_stop(cvImg)
        
        # 12. Попытка остановиться (финиш)
        if self.TASK_LEVEL == 2 and (time.time() - self.START_TIME) > 15:
            self._msg.data = "autorace_data"
            self._sign_finish.publish(self._msg)   
        
        # 13. Управление роботом (только если не остановлен)
        if not self._is_stopped:
            if abs(error) > OFFSET_BTW_CENTERS:
                # Преобразуем ошибку в угол
                angle_error = normalized_error * 0.5
                
                angular_v = self._compute_PID(angle_error)
                self.twist.angular.z = angular_v

                if ANALOG_CAP_MODE:
                    # Плавное уменьшение скорости при поворотах
                    speed_factor = max(0.4, 1.0 - abs(angular_v) / MAXIMUM_ANGLUAR_SPEED_CAP)
                    self.twist.linear.x = self._linear_speed * speed_factor
            else:
                # Если ошибка маленькая, едем прямо
                self.twist.angular.z = 0.0
                self.twist.linear.x = self._linear_speed

        # 14. Визуализация
        if INFO_LEVEL:
            # Создаем изображение для отладки
            debug_img = perspective.copy()
            
            # Добавляем информацию о состоянии остановки
            if self._is_stopped:
                cv2.putText(debug_img, "STOPPED", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Рисуем линию сканирования
            cv2.line(debug_img, (0, hLevelLine), (persective_w, hLevelLine), (255, 0, 0), 2)
            
            # Рисуем края линий
            cv2.line(debug_img, (int(yellow_edge), hLevelLine-30), (int(yellow_edge), hLevelLine+30), 
                    (0, 255, 255), 5)  # Желтая линия
            cv2.line(debug_img, (int(white_edge), hLevelLine-30), (int(white_edge), hLevelLine+30), 
                    (255, 255, 255), 5)  # Белая линия
            
            # Рисуем центры
            center_crds = (int(image_center), hLevelLine)
            lines_center_crds = (int(road_center), hLevelLine)
            
            cv2.circle(debug_img, center_crds, 8, (0, 255, 0), -1)  # Центр изображения - зеленый
            if self.point_status:
                cv2.circle(debug_img, lines_center_crds, 8, (0, 0, 255), -1)  # Центр дороги - красный
            else:
                cv2.circle(debug_img, lines_center_crds, 8, (100, 100, 100), -1)  # Центр дороги - серый
            
            # Рисуем линию между центрами
            cv2.line(debug_img, center_crds, lines_center_crds, (255, 0, 255), 3)
            
            # Отображаем информацию
            cv2.putText(debug_img, f"Error: {error:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Yellow: {yellow_edge:.0f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_img, f"White: {white_edge:.0f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Road Center: {road_center:.0f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(debug_img, f"Status: {'OK' if self.point_status else 'LOST'}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.point_status else (0, 0, 255), 2)
            cv2.putText(debug_img, f"Task Level: {self.TASK_LEVEL}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Main Line: {self.MAIN_LINE}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Angle: {angle:.2f} rad", (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"At Intersection: {'YES' if is_at_intersection else 'NO'}", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Ignore Stop: {self._ignore_stop_sign}", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_img, f"Dir Determined: {self._direction_determined}", (10, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Road Following Debug", debug_img)
            cv2.waitKey(1)

        # 15. Публикуем команды управления
        if self.STATUS_CAR == 1 and not self._is_stopped:
            self._robot_cmd_vel_pub.publish(self.twist)


def main():
    rclpy.init()
    FTN = Follow_Trace_Node()
    rclpy.spin(FTN)
    FTN.destroy_node()
    rclpy.shutdown()