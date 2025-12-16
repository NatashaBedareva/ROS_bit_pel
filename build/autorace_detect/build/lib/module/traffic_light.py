import cv2
import numpy as np

from module.logger import log_info

# Проверка зеленого цвета на экране
def check_green_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    return cv2.countNonZero(green_mask) > 0

# Проверка готовности к движению, если был найден зеленый цвет, то меняем статусы
def check_traffic_lights(follow_trace, img):
    is_green_present = check_green_color(img)

    if is_green_present:
        log_info(follow_trace, "Поехали", msg_id=2)
        follow_trace.STATUS_CAR = 1
        follow_trace.TASK_LEVEL = 1
    else:
        log_info(follow_trace, "Ждём зеленый сигнал", 0)