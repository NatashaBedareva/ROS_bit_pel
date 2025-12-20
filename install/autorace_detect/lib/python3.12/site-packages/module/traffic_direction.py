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
def check_stop_sign(img):
    """
    Проверяет наличие знака остановки.
    """
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

# Функция для выделения знака на изображении
def extract_sign_region(img):
    """
    Выделяет область с дорожным знаком для анализа.
    Возвращает маску знака и координаты ограничивающего прямоугольника.
    """
    blue_mask = check_blue_color(img)
    height, width = blue_mask.shape[:2]
    
    # Находим контуры в синей маске
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Находим самый большой контур (скорее всего, это знак)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Проверяем размер контура - должен быть достаточно большим
    area = cv2.contourArea(largest_contour)
    if area < 500:  # Минимальная площадь знака
        return None, None
    
    # Получаем ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Создаем маску только для этого контура
    sign_mask = np.zeros_like(blue_mask)
    cv2.drawContours(sign_mask, [largest_contour], -1, 255, -1)
    
    # Вырезаем область знака из исходного изображения
    sign_roi = sign_mask[y:y+h, x:x+w]
    
    if INFO_LEVEL:
        # Показываем выделенный знак
        sign_img = img[y:y+h, x:x+w].copy()
        cv2.imshow("Extracted Sign", sign_img)
        cv2.waitKey(1)
    
    return sign_mask, (x, y, w, h)

# Анализ направления стрелки на знаке
def analyze_arrow_direction(img):
    """
    Анализирует направление стрелки на знаке.
    Возвращает: "LEFT", "RIGHT", "STRAIGHT", или None
    """
    # Сначала выделяем знак
    sign_mask, bbox = extract_sign_region(img)
    
    if sign_mask is None or bbox is None:
        return None
    
    x, y, w, h = bbox
    
    # Создаем копию изображения для анализа
    analysis_img = img.copy()
    
    # Разделяем знак на три вертикальные зоны
    zone_width = w // 3
    left_zone = sign_mask[y:y+h, x:x+zone_width]
    center_zone = sign_mask[y:y+h, x+zone_width:x+2*zone_width]
    right_zone = sign_mask[y:y+h, x+2*zone_width:x+w]
    
    # Считаем синие пиксели в каждой зоне
    left_pixels = cv2.countNonZero(left_zone)
    center_pixels = cv2.countNonZero(center_zone)
    right_pixels = cv2.countNonZero(right_zone)
    
    # Разделяем знак на две горизонтальные зоны (верх/низ)
    zone_height = h // 2
    top_zone = sign_mask[y:y+zone_height, x:x+w]
    bottom_zone = sign_mask[y+zone_height:y+h, x:x+w]
    
    top_pixels = cv2.countNonZero(top_zone)
    bottom_pixels = cv2.countNonZero(bottom_zone)
    
    # Анализируем форму стрелки
    # Для стрелки влево: больше пикселей в правой части
    # Для стрелки вправо: больше пикселей в левой части
    # Для стрелки прямо: симметричное распределение
    
    # Вычисляем соотношения
    left_right_ratio = left_pixels / right_pixels if right_pixels > 0 else 100
    right_left_ratio = right_pixels / left_pixels if left_pixels > 0 else 100
    
    # Визуализация для отладки
    if INFO_LEVEL:
        debug_img = img.copy()
        
        # Рисуем ограничивающий прямоугольник знака
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Рисуем разделительные линии
        cv2.line(debug_img, (x+zone_width, y), (x+zone_width, y+h), (255, 0, 0), 1)
        cv2.line(debug_img, (x+2*zone_width, y), (x+2*zone_width, y+h), (255, 0, 0), 1)
        cv2.line(debug_img, (x, y+zone_height), (x+w, y+zone_height), (255, 255, 0), 1)
        
        # Отображаем информацию
        cv2.putText(debug_img, f"Left: {left_pixels}", (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Center: {center_pixels}", (x+zone_width, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(debug_img, f"Right: {right_pixels}", (x+2*zone_width, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Определяем направление
        direction = None
        color = (128, 128, 128)
        
        if right_pixels > left_pixels * 1.5 and right_pixels > center_pixels:
            direction = "LEFT"  # Стрелка влево: пиксели справа (острие стрелки)
            color = (0, 255, 0)
            direction_text = "ARROW: TURN LEFT"
        elif left_pixels > right_pixels * 1.5 and left_pixels > center_pixels:
            direction = "RIGHT"  # Стрелка вправо: пиксели слева (острие стрелки)
            color = (0, 0, 255)
            direction_text = "ARROW: TURN RIGHT"
        elif center_pixels > left_pixels and center_pixels > right_pixels:
            direction = "STRAIGHT"  # Стрелка прямо: пиксели в центре
            color = (255, 255, 0)
            direction_text = "ARROW: GO STRAIGHT"
        else:
            direction_text = "ARROW: UNKNOWN"
        
        cv2.putText(debug_img, direction_text, (x, y-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Arrow Direction Analysis", debug_img)
        cv2.waitKey(1)
    
    # Определяем направление стрелки
    # Логика: стрелка указывает НАПРАВО, если острие справа (больше синего слева)
    #         стрелка указывает НАЛЕВО, если острие слева (больше синего справа)
    if right_pixels > left_pixels * 1.5 and right_pixels > center_pixels:
        return "LEFT"  # Острие справа → стрелка указывает налево
    elif left_pixels > right_pixels * 1.5 and left_pixels > center_pixels:
        return "RIGHT"  # Острие слева → стрелка указывает направо
    elif center_pixels > left_pixels and center_pixels > right_pixels:
        return "STRAIGHT"  # Стрелка прямо
    else:
        return None

# Основная функция проверки направления
def check_direction(follow_trace, img, is_at_intersection=False, ignore_stop_sign=False):
    """
    Проверяет направление на перекрестке.
    
    Args:
        follow_trace: объект Follow_Trace_Node
        img: изображение с камеры
        is_at_intersection: если True, робот находится на перекрестке
        ignore_stop_sign: если True, игнорируем проверку знака STOP
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
        cv2.putText(display_img, f"Ignore Stop: {ignore_stop_sign}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if abs(angle) >= 2.80 and follow_trace.TASK_LEVEL == 1:
            cv2.putText(display_img, "AT INTERSECTION - ANALYZING", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Robot Camera View", display_img)
        cv2.waitKey(1)
    
    # 1. Проверяем знак остановки (если на перекрестке и не игнорируем)
    if is_at_intersection and follow_trace.TASK_LEVEL == 1 and not ignore_stop_sign:
        if check_stop_sign(img):
            log_info(follow_trace, "[ДОРОЖНЫЙ ЗНАК] Обнаружен знак остановки")
            return "STOP_SIGN"  # Возвращаем сигнал об остановке
    
    # 2. Если на перекрестке и TASK_LEVEL == 1, проверяем направление
    if is_at_intersection and follow_trace.TASK_LEVEL == 1:
        # Анализируем направление стрелки на знаке
        direction = analyze_arrow_direction(img)
        
        if direction == "LEFT":
            log_info(follow_trace, "[Перекресток] Стрелка указывает: ПОВОРОТ НАЛЕВО")
            follow_trace.MAIN_LINE = "YELLOW"
        elif direction == "RIGHT":
            log_info(follow_trace, "[Перекресток] Стрелка указывает: ПОВОРОТ НАПРАВО")
            follow_trace.MAIN_LINE = "WHITE"
        elif direction == "STRAIGHT":
            log_info(follow_trace, "[Перекресток] Стрелка указывает: ПРЯМО")
            follow_trace.MAIN_LINE = "WHITE"  # По умолчанию едем по белой линии
        else:
            # Если не определили направление стрелки, используем старую логику
            log_info(follow_trace, "[Перекресток] Направление не определено, поворачиваю налево")
            follow_trace.MAIN_LINE = "YELLOW"
        
        follow_trace.TASK_LEVEL = 2
        return "DIRECTION_DECIDED"
    
    return None  # Ничего не обнаружено