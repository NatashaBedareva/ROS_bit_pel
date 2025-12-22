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

def analyze_arrow_direction(img):
    """
    Анализирует направление стрелки на знаке.
    Темно-серая/белая стрелка на синем фоне.
    Возвращает: "LEFT", "RIGHT", или None
    """
    # Сначала выделяем синий знак
    sign_mask, bbox = extract_sign_region(img)
    
    if sign_mask is None or bbox is None:
        return None
    
    x, y, w, h = bbox
    
    # Вырезаем область знака из исходного изображения
    sign_roi = img[y:y+h, x:x+w].copy()
    
    # Преобразуем в RGB для более простой работы
    # Ищем серый цвет: R ≈ G ≈ B, и значения около 93
    rgb_roi = sign_roi.copy()
    
    # Вычисляем разницу между каналами
    r_channel = rgb_roi[:, :, 2].astype(np.float32)
    g_channel = rgb_roi[:, :, 1].astype(np.float32)
    b_channel = rgb_roi[:, :, 0].astype(np.float32)
    
    # Маска для серого цвета: все каналы примерно равны
    diff_rg = np.abs(r_channel - g_channel)
    diff_rb = np.abs(r_channel - b_channel)
    diff_gb = np.abs(g_channel - b_channel)
    
    # Серый цвет: разница между каналами не более 30
    gray_mask1 = (diff_rg < 30) & (diff_rb < 30) & (diff_gb < 30)
    
    # Также проверяем яркость: R+G+B примерно 93*3 ≈ 280
    brightness = r_channel + g_channel + b_channel
    gray_mask2 = (brightness > 200) & (brightness < 350)  # Примерно 93 ± 50
    
    # Объединяем маски
    gray_mask = gray_mask1 & gray_mask2
    
    # Преобразуем в uint8 для OpenCV
    gray_mask_uint8 = gray_mask.astype(np.uint8) * 255
    
    # Применяем маску знака (только внутри синей области)
    sign_binary = sign_mask[y:y+h, x:x+w]  # Уже нужный срез
    gray_in_sign = cv2.bitwise_and(gray_mask_uint8, sign_binary)
    
    # Морфологические операции для очистки маски стрелки
    kernel = np.ones((3, 3), np.uint8)
    gray_in_sign = cv2.morphologyEx(gray_in_sign, cv2.MORPH_OPEN, kernel)
    gray_in_sign = cv2.morphologyEx(gray_in_sign, cv2.MORPH_CLOSE, kernel)
    
    # Проверяем, достаточно ли серых пикселей для стрелки
    gray_pixels = cv2.countNonZero(gray_in_sign)
    total_sign_pixels = cv2.countNonZero(sign_binary)
    
    if total_sign_pixels == 0 or gray_pixels < 20:  # Минимум 20 серых пикселей
        return None
    
    # Альтернативный подход: использовать HSV и искать цвета с низкой насыщенностью
    hsv_roi = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2HSV)
    
    # Для серого/белого: низкая насыщенность (S < 50), средняя/высокая яркость
    lower_gray = np.array([0, 0, 70])  # H: любой, S: низкая, V: средняя
    upper_gray = np.array([180, 50, 200])  # H: любой, S: низкая, V: высокая
    
    hsv_gray_mask = cv2.inRange(hsv_roi, lower_gray, upper_gray)
    hsv_gray_in_sign = cv2.bitwise_and(hsv_gray_mask, sign_binary)
    
    # Объединяем оба метода для лучшего результата
    combined_gray = cv2.bitwise_or(gray_in_sign, hsv_gray_in_sign)
    
    # ПРОСТОЙ ПОДХОД: делим знак пополам и смотрим, где больше серого
    middle_x = w // 2
    
    # Левая половина (ИСПРАВЛЕНО: правильно определяем границы)
    left_half = combined_gray[:, :middle_x]  # От 0 до middle_x
    # Правая половина  
    right_half = combined_gray[:, middle_x:]  # От middle_x до конца
    
    # Считаем серые пиксели в каждой половине
    left_pixels = cv2.countNonZero(left_half)
    right_pixels = cv2.countNonZero(right_half)
    
    # Вычисляем соотношение
    total_gray = left_pixels + right_pixels
    if total_gray == 0:
        return None
    
    left_ratio = left_pixels / total_gray
    right_ratio = right_pixels / total_gray
    
    # Визуализация для отладки
    if INFO_LEVEL:
        debug_img = img.copy()
        
        # Рисуем ограничивающий прямоугольник знака
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Рисуем вертикальную линию посередине
        cv2.line(debug_img, (x+middle_x, y), (x+middle_x, y+h), (255, 0, 0), 2)
        
        # Показываем маску серой стрелки на изображении
        overlay_roi = debug_img[y:y+h, x:x+w]
        # Создаем цветную версию маски для наложения
        colored_gray = cv2.cvtColor(combined_gray, cv2.COLOR_GRAY2BGR)
        colored_gray[combined_gray > 0] = [200, 200, 200]  # Серый цвет
        # Накладываем с прозрачностью
        cv2.addWeighted(colored_gray, 0.5, overlay_roi, 0.5, 0, overlay_roi)
        
        # Отображаем информацию
        cv2.putText(debug_img, f"Gray: {gray_pixels} | Total: {total_gray}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Left: {left_pixels} ({left_ratio:.1%})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Right: {right_pixels} ({right_ratio:.1%})", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Показываем средний цвет в области знака для отладки
        if gray_pixels > 0:
            # Получаем координаты серых пикселей
            gray_coords = np.where(combined_gray > 0)
            if len(gray_coords[0]) > 0:
                avg_color = np.mean(sign_roi[gray_coords], axis=0)
                cv2.putText(debug_img, f"Avg color: B:{avg_color[0]:.0f} G:{avg_color[1]:.0f} R:{avg_color[2]:.0f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Определяем направление
        direction = None
        color = (128, 128, 128)
        direction_text = "ARROW: UNKNOWN"
        
        # Простая логика: где больше серого, туда и поворачивать
        threshold_ratio = 1.3  # На 30% больше
        
        if left_pixels > right_pixels * threshold_ratio:
            direction = "LEFT"
            color = (0, 255, 0)  # Зеленый
            direction_text = "ARROW: TURN LEFT"
        elif right_pixels > left_pixels * threshold_ratio:
            direction = "RIGHT"
            color = (0, 0, 255)  # Красный
            direction_text = "ARROW: TURN RIGHT"
        else:
            # Если разница небольшая, но есть явное преимущество
            if left_pixels > right_pixels + 15:  # Разница хотя бы 15 пикселей
                direction = "LEFT"
                color = (0, 255, 0)
                direction_text = "ARROW: TURN LEFT (slight)"
            elif right_pixels > left_pixels + 15:
                direction = "RIGHT"
                color = (0, 0, 255)
                direction_text = "ARROW: TURN RIGHT (slight)"
        
        cv2.putText(debug_img, direction_text, (x, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Также показываем только знак с выделенной стрелкой - ИСПРАВЛЕННАЯ ЧАСТЬ
        sign_debug = sign_roi.copy()
        
        # Создаем цветные маски для левой и правой половин
        left_colored = np.zeros_like(sign_roi)
        right_colored = np.zeros_like(sign_roi)
        
        # Получаем индексы пикселей в каждой половине
        left_indices = np.where(left_half > 0)
        right_indices = np.where(right_half > 0)
        
        # Раскрашиваем пиксели
        if len(left_indices[0]) > 0:
            # Преобразуем индексы для правой половины (учитываем смещение)
            left_rows = left_indices[0]
            left_cols = left_indices[1]
            sign_debug[left_rows, left_cols] = [0, 255, 0]  # Зеленый
        
        if len(right_indices[0]) > 0:
            right_rows = right_indices[0]
            right_cols = right_indices[1]
            # Для правой половины нужно добавить middle_x к столбцам
            sign_debug[right_rows, right_cols + middle_x] = [0, 0, 255]  # Красный
        
        # Рисуем линию посередине на знаке
        cv2.line(sign_debug, (middle_x, 0), (middle_x, h), (255, 255, 255), 2)
        
        cv2.imshow("Arrow Direction Analysis", debug_img)
        cv2.imshow("Sign with Gray Arrow", sign_debug)
        cv2.waitKey(1)
    
    # ПРОСТАЯ ЛОГИКА: где больше серого, туда и поворачивать
    # Если больше серого в ЛЕВОЙ половине → поворот НАЛЕВО
    # Если больше серого в ПРАВОЙ половине → поворот НАПРАВО
    
    # Пороговое значение
    threshold_ratio = 1.3  # На 30% больше
    
    if left_pixels > right_pixels * threshold_ratio:
        return "RIGHT"  # Больше серого слева → поворачиваем налево
    elif right_pixels > left_pixels * threshold_ratio:
        return "LEFT"  # Больше серого справа → поворачиваем направо
    else:
        # Если разница небольшая, но есть явное преимущество
        if left_pixels > right_pixels + 15:  # Разница хотя бы 15 пикселей
            return "RIGHT"
        elif right_pixels > left_pixels + 15:
            return "LEFT"
    
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
        else:
            # Если не определили направление стрелки, используем старую логику
            log_info(follow_trace, "[Перекресток] Направление не определено, поворачиваю налево")
            follow_trace.MAIN_LINE = "YELLOW"
        
        follow_trace.TASK_LEVEL = 2
        return "DIRECTION_DECIDED"
    
    return None  # Ничего не обнаружено