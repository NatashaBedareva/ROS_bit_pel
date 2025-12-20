
import cv2
import numpy as np
from module.config import INFO_LEVEL
from module.logger import log_info

# Определение синего цвета в HSV (оптимизированный диапазон)
def check_blue_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Диапазон для синего цвета дорожных знаков
    lower_blue = np.array([100, 100, 50])  
    upper_blue = np.array([130, 255, 255])  
    
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    # Морфологические операции для очистки маски
    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    return blue_mask 

# Проверка на знак остановки (возвращает True если виден синий знак)
def check_stop_sign(img, already_stopped=False):
    """
    Проверяет наличие знака остановки.
    
    Args:
        img: изображение с камеры
        already_stopped: если True, не проверяем знак (робот уже остановлен)
    """
    if already_stopped:
        return False  # Если уже остановлен, не проверяем снова
    
    blue_mask = check_blue_color(img)
    
    # Анализируем распределение синих пикселей
    height, width = blue_mask.shape[:2]
    
    # ВЕРХНЯЯ область для обнаружения знаков - СМЕЩЕНА ВЛЕВО
    roi_height = height // 4
    roi_width = width // 3
    roi_y = height // 8
    roi_x = width // 4  # СМЕЩАЕМ ВЛЕВО: было (width - roi_width) // 2, стало width // 4
    
    # Проверяем границы
    if roi_x + roi_width > width:
        roi_x = width - roi_width
    if roi_x < 0:
        roi_x = 0
    
    roi = blue_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    blue_pixels = cv2.countNonZero(roi)
    
    # Считаем соотношение синих пикселей
    total_pixels_roi = roi_height * roi_width
    blue_ratio = blue_pixels / total_pixels_roi if total_pixels_roi > 0 else 0
    
    # Детектируем круглые объекты в синей маске
    circles_detected = False
    if blue_pixels > 100:
        roi_contours = blue_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width].copy()
        contours, _ = cv2.findContours(roi_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circles_detected = circularity > 0.6
    
    # Пороги для знака остановки
    stop_threshold_pixels = 300
    stop_threshold_ratio = 0.15
    
    if INFO_LEVEL:
        debug_img = img.copy()
        # Рисуем ROI область (зеленая рамка) - СМЕЩЕНА ВЛЕВО
        cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)
        
        if circles_detected:
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 0, 255), 3)
        
        cv2.putText(debug_img, f"Blue in ROI: {blue_pixels}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"ROI Position: ({roi_x},{roi_y})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        blue_mask_colored = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        blue_mask_colored[:, :, 0] = 255
        blue_mask_colored[:, :, 1] = 0
        blue_mask_colored[:, :, 2] = 0
        
        alpha = 0.3
        overlay = cv2.addWeighted(debug_img, 1, blue_mask_colored, alpha, 0)
        
        is_stop_sign = (blue_pixels > stop_threshold_pixels and 
                       blue_ratio > stop_threshold_ratio and
                       circles_detected)
        
        if is_stop_sign:
            cv2.putText(overlay, "STOP SIGN DETECTED!", (width//2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow("Stop Sign Detection", overlay)
        cv2.waitKey(1)
    
    return (blue_pixels > stop_threshold_pixels and 
            blue_ratio > stop_threshold_ratio and
            circles_detected)

# Определение направления поворота на знаке
def check_direction(follow_trace, img, already_stopped=False):
    """
    Проверяет направление на перекрестке.
    
    Args:
        follow_trace: объект Follow_Trace_Node
        img: изображение с камеры
        already_stopped: если True, пропускаем проверку знака остановки
    """
    angle = follow_trace.get_angle()
    
    # Показываем, что видит робот перед собой
    if INFO_LEVEL:
        display_img = img.copy()
        height, width = display_img.shape[:2]
        
        # Линия для анализа направления - СМЕЩЕНА ВЛЕВО
        analysis_line_x = 200  # БЫЛО 300, СМЕЩАЕМ ВЛЕВО
        cv2.line(display_img, (width//2, 0), (width//2, height), (0, 255, 0), 1)
        cv2.line(display_img, (analysis_line_x, 0), (analysis_line_x, height), (0, 200, 0), 1)
        
        cv2.putText(display_img, f"Angle: {angle:.2f} rad", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Task Level: {follow_trace.TASK_LEVEL}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Analysis Line: x={analysis_line_x}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if abs(angle) >= 2.80 and follow_trace.TASK_LEVEL == 1:
            cv2.putText(display_img, "AT INTERSECTION - ANALYZING", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Robot Camera View", display_img)
        cv2.waitKey(1)
    
    # 1. Проверяем знак остановки только если НЕ уже остановлены
    if follow_trace.TASK_LEVEL == 1 and not already_stopped:
        if check_stop_sign(img, already_stopped):
            log_info(follow_trace, "[ДОРОЖНЫЙ ЗНАК] Обнаружен знак остановки")
            return "STOP"  # Возвращаем сигнал об остановке
    
    # 2. Проверяем направление поворота (только если робот на перекрестке и TASK_LEVEL == 1)
    if abs(angle) >= 2.80 and follow_trace.TASK_LEVEL == 1:
        blue_mask = check_blue_color(img)
        
        # Фокус на верхнюю часть для знаков направления
        height, width = blue_mask.shape[:2]
        upper_part = height // 3
        blue_mask_upper = blue_mask[:upper_part, :]
        
        # СМЕЩАЕМ ОБЛАСТЬ АНАЛИЗА ВЛЕВО
        analysis_line_x = 200  # БЫЛО 300
        
        if INFO_LEVEL:
            overlay = img.copy()
            cv2.line(overlay, (0, upper_part), (width, upper_part), (255, 255, 0), 2)
            
            blue_mask_colored = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
            blue_mask_colored[:, :, 0] = 255
            
            alpha = 0.3
            cv2.addWeighted(blue_mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Рисуем разделительную линию - СМЕЩЕНА ВЛЕВО
            cv2.line(overlay, (analysis_line_x, 0), (analysis_line_x, height), (0, 255, 0), 2)
            
            left_half_upper = blue_mask_upper[:, :analysis_line_x]
            right_half_upper = blue_mask_upper[:, analysis_line_x:]
            left_count = cv2.countNonZero(left_half_upper)
            right_count = cv2.countNonZero(right_half_upper)
            
            cv2.putText(overlay, f"L: {left_count}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"R: {right_count}", (analysis_line_x + 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Blue Detection", overlay)
            cv2.waitKey(1)
        
        # Используем смещенную линию для анализа
        left_half = blue_mask_upper[:, :analysis_line_x]
        right_half = blue_mask_upper[:, analysis_line_x:]

        if INFO_LEVEL:
            direction_img = img.copy()
            height, width = direction_img.shape[:2]
            
            left_count = cv2.countNonZero(left_half)
            right_count = cv2.countNonZero(right_half)
            
            cv2.putText(direction_img, f"Left (upper): {left_count}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(direction_img, f"Right (upper): {right_count}", (analysis_line_x + 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if left_count >= right_count:
                result_text = "DECISION: TURN LEFT"
                color = (0, 255, 0)
            else:
                result_text = "DECISION: TURN RIGHT"
                color = (0, 0, 255)
            
            cv2.putText(direction_img, result_text, (width//2 - 150, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            cv2.imshow("Direction Decision", direction_img)
            cv2.waitKey(1)

        # Определяем направление
        if cv2.countNonZero(left_half) >= cv2.countNonZero(right_half):
            log_info(follow_trace, "[Перекресток] Поворот налево")
            follow_trace.MAIN_LINE = "YELLOW"
        else:
            log_info(follow_trace, "[Перекресток] Поворот направо")
            follow_trace.MAIN_LINE = "WHITE"
        
        follow_trace.TASK_LEVEL = 2
        return "TURN"
    
    return None  # Ничего не обнаружено
