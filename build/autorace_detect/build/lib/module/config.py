from typing import Literal

OFFSET_BTW_CENTERS : int = 20 # Увеличил смещение для более плавного реагирования
LINES_H_RATIO : float = 0.7  # Использую 70% высоты - хороший компромисс
MAX_LINIEAR_SPEED : float = 0.08  # Чуть уменьшил скорость для стабильности
ANALOG_CAP_MODE : bool = True
MAXIMUM_ANGLUAR_SPEED_CAP : float = 1.5

FOLLOW_ROAD_CROP_HALF  : Literal[True, False] = False
FOLLOW_ROAD_MODE : Literal["BOTH", "YELLOW", "WHITE"] = "BOTH"
WHITE_MODE_CONSTANT : int = 500  # Увеличил константы
YELLOW_MODE_CONSTANT : int = 150
LINE_HISTORY_SIZE : int = 10  # Увеличил историю

INFO_LEVEL : Literal[True, False] = True
STATUS_CAR : Literal[0, 1] = 0
LOGGING_POOL_SIZE : int = 3




STOP_SIGN_ENABLED: bool = True
STOP_SIGN_DURATION: float = 3.0  # Длительность остановки в секундах
STOP_SIGN_CHECK_INTERVAL: float = 0.5  # Интервал проверки знака
STOP_SIGN_THRESHOLD: int = 1000  # Порог синих пикселей для знака остановки
