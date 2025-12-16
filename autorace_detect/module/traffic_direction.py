import cv2
import numpy as np

from module.logger import log_info

# Определение синего цвета в HSV
def check_blue_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 40, 40])  
    upper_blue = np.array([255, 255, 255])  
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    return blue_mask 

# Определение направления поворота на знаке
def check_direction(follow_trace, img):
    angle = follow_trace.get_angle()
    if abs(angle) >= 2.80:
        blue_mask = check_blue_color(img)
        
        left_half = blue_mask[:,:300]
        right_half = blue_mask[:,300:]

        if cv2.countNonZero(left_half) >= cv2.countNonZero(right_half):
            log_info(follow_trace, "[Перекресток] Поворот налево")
            follow_trace.MAIN_LINE = "YELLOW"
        else:
            log_info(follow_trace, "[Перекресток] Поворот направо")
            follow_trace.MAIN_LINE = "WHITE"
        
        follow_trace.TASK_LEVEL = 2
