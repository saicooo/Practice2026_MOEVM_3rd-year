# main.py
import numpy as np


def controller(control_drones):
    # Целевые точки
    TARGET_HEIGHT = 2.0  # высота 2 метра
    TARGET_Y = 2.0       # движение по Y на 2 метра
    TARGET_X = 5.0       # движение по X на 5 метров
    
    # Параметры PID-регуляторов
    KP_POS = 0.8      # коэффициент для позиции
    KP_VEL = 0.3      # коэффициент для скорости
    KI = 0.05         # интегральный коэффициент
    KD = 0.2          # дифференциальный коэффициент
    
    # Базовые обороты для зависания
    base_rpm = 14468.429
    
    # Состояние автомата
    STATE_TAKEOFF = 0    # взлет
    STATE_MOVE_Y = 1     # движение по Y
    STATE_MOVE_X = 2     # движение по X
    STATE_LANDING = 3    # посадка
    current_state = STATE_TAKEOFF
    
    # Переменные для PID
    prev_errors = [0.0, 0.0, 0.0]  # z, y, x
    integral_errors = [0.0, 0.0, 0.0]
    prev_time = None
    
    # Скорость вращения винтов
    action = np.array([[base_rpm, base_rpm, base_rpm, base_rpm]])
    
    while True:
        xyz_pos, xyz_speed = control_drones(action)
        
        # Получаем текущее время (имитируем, используя итерации)
        # В реальном времени здесь был бы time.time()
        # Для симуляции используем счетчик итераций как время
        
        # Целевая позиция в зависимости от состояния
        target_x, target_y, target_z = 0, 0, 0
        
        if current_state == STATE_TAKEOFF:
            target_z = TARGET_HEIGHT
            # Переход к следующему состоянию при достижении высоты
            if xyz_pos[0, 2] >= TARGET_HEIGHT - 0.05:
                current_state = STATE_MOVE_Y
                # Сброс интегральных ошибок при смене состояния
                integral_errors = [0.0, 0.0, 0.0]
                prev_errors = [0.0, 0.0, 0.0]
                
        elif current_state == STATE_MOVE_Y:
            target_z = TARGET_HEIGHT
            target_y = TARGET_Y
            # Переход к следующему состоянию
            if abs(xyz_pos[0, 1] - TARGET_Y) < 0.05:
                current_state = STATE_MOVE_X
                integral_errors = [0.0, 0.0, 0.0]
                prev_errors = [0.0, 0.0, 0.0]
                
        elif current_state == STATE_MOVE_X:
            target_z = TARGET_HEIGHT
            target_y = TARGET_Y
            target_x = TARGET_X
            # Переход к посадке
            if abs(xyz_pos[0, 0] - TARGET_X) < 0.05:
                current_state = STATE_LANDING
                integral_errors = [0.0, 0.0, 0.0]
                prev_errors = [0.0, 0.0, 0.0]
                
        elif current_state == STATE_LANDING:
            target_z = 0
            target_y = TARGET_Y
            target_x = TARGET_X
            # Завершаем при посадке
            if xyz_pos[0, 2] < 0.05:
                break
        
        # Вычисляем ошибки по осям
        errors = [
            target_z - xyz_pos[0, 2],  # ошибка по Z
            target_y - xyz_pos[0, 1],  # ошибка по Y
            target_x - xyz_pos[0, 0]   # ошибка по X
        ]
        
        # Обновление интегральной ошибки
        for i in range(3):
            integral_errors[i] += errors[i] * 0.01  # dt примерно 0.01
        
        # Вычисление производной ошибки
        derivatives = [errors[i] - prev_errors[i] for i in range(3)]
        
        # PID-регуляторы для каждой оси
        # Преобразуем ошибки в управляющие сигналы
        control_z = KP_POS * errors[0] + KI * integral_errors[0] + KD * derivatives[0]
        control_y = KP_POS * errors[1] + KI * integral_errors[1] + KD * derivatives[1]
        control_x = KP_POS * errors[2] + KI * integral_errors[2] + KD * derivatives[2]
        
        # Ограничение управляющих сигналов
        max_control = 500
        control_z = max(-max_control, min(max_control, control_z))
        control_y = max(-max_control, min(max_control, control_y))
        control_x = max(-max_control, min(max_control, control_x))
        
        # Сохранение ошибок для следующей итерации
        prev_errors = errors.copy()
        
        # Преобразование управляющих сигналов в обороты винтов
        # Схема расположения винтов:
        # 2 (верхний левый) и 0 (нижний правый) управляют креном по X
        # 1 (нижний левый) и 3 (верхний правый) управляют креном по Y
        # Суммарное изменение для тангажа и крена
        
        # Балансировка: изменение оборотов для движения
        rpm_0 = base_rpm - control_x - control_y + control_z  # правый нижний
        rpm_1 = base_rpm + control_x - control_y + control_z  # левый нижний
        rpm_2 = base_rpm - control_x + control_y + control_z  # левый верхний
        rpm_3 = base_rpm + control_x + control_y + control_z  # правый верхний
        
        # Ограничение оборотов
        min_rpm = 10000
        max_rpm = 20000
        rpm_0 = max(min_rpm, min(max_rpm, rpm_0))
        rpm_1 = max(min_rpm, min(max_rpm, rpm_1))
        rpm_2 = max(min_rpm, min(max_rpm, rpm_2))
        rpm_3 = max(min_rpm, min(max_rpm, rpm_3))
        
        action = np.array([[rpm_0, rpm_1, rpm_2, rpm_3]])
    
    return action