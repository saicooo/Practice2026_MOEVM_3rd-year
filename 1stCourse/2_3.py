# main.py
import numpy as np
from PIL import Image
import cv2


def controller(control_drones):
    action = np.array([0, 0, 0])
    
    # Состояния
    CENTER = 0
    HOVER = 1
    NEXT = 2
    
    state = CENTER
    hover_steps = 0
    step = 0
    
    # Отслеживание позиции
    estimated_pos = np.array([0.0, 0.0])  # относительная позиция
    visited = [(0.0, 0.0)]  # посещенные точки
    
    while True:
        image = control_drones([action])[0]
        step += 1
        
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        center = np.array([w/2, h/2])
        
        # Простое и быстрое обнаружение
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Красный
        red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        red_points = np.argwhere(red_mask > 0)
        
        # Синий  
        blue_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([140, 255, 255]))
        blue_points = np.argwhere(blue_mask > 0)
        
        has_red = len(red_points) > 50
        has_blue = len(blue_points) > 30
        
        if state == CENTER:
            if has_red:
                red_center = np.mean(red_points[:, ::-1], axis=0)  # (x, y)
                error = red_center - center
                
                if np.linalg.norm(error) < 15:
                    state = HOVER
                    hover_steps = 0
                    action = np.array([0, 0, 0])
                else:
                    correction = error * 0.008
                    correction[1] *= -1
                    action = np.array([correction[0], correction[1], 0])
                    action = np.clip(action, -0.6, 0.6)
            else:
                # Ищем красный
                action = np.array([0.3, 0.3, 0])
                
        elif state == HOVER:
            if hover_steps < 2:
                if has_red:
                    red_center = np.mean(red_points[:, ::-1], axis=0)
                    error = red_center - center
                    if np.linalg.norm(error) > 20:
                        correction = error * 0.005
                        correction[1] *= -1
                        action = np.array([correction[0], correction[1], 0])
                        action = np.clip(action, -0.3, 0.3)
                    else:
                        action = np.array([0, 0, 0])
                else:
                    action = np.array([0, 0, 0])
                hover_steps += 1
            else:
                state = NEXT
                action = np.array([0, 0, 0])
                
        elif state == NEXT:
            if has_blue:
                blue_center = np.mean(blue_points[:, ::-1], axis=0)
                direction = blue_center - center
                
                if has_red:
                    red_center = np.mean(red_points[:, ::-1], axis=0)
                    # Направление от красного к синему
                    path_direction = blue_center - red_center
                    if np.linalg.norm(path_direction) > 0:
                        path_direction = path_direction / np.linalg.norm(path_direction)
                else:
                    if np.linalg.norm(direction) > 0:
                        path_direction = direction / np.linalg.norm(direction)
                    else:
                        path_direction = np.array([1, 0])
                
                # Двигаемся по направлению
                speed = 0.5
                movement = path_direction * speed
                action = np.array([movement[0], -movement[1], 0])
                
                # Обновляем оценку позиции
                estimated_pos += movement * 0.75  # примерно 0.75 секунды на шаг
            else:
                # Исследуем
                if step % 15 < 5:
                    action = np.array([0.4, 0, 0])
                elif step % 15 < 10:
                    action = np.array([0, 0.4, 0])
                else:
                    action = np.array([-0.4, -0.4, 0])
                
                if has_red:
                    state = CENTER
        
        action = np.clip(action, -1, 1)