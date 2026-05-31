def is_passable(x, y, env, grid_size):
    # Проверка границ поля
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        return False

    cell = env.get_cell(x, y)

    # Пустая клетка
    if cell is None:
        return True

    # Стена
    if cell.type == "wall":
        return False

    # Препятствие
    if cell.type == "obstacle":
        return False

    return True


def solution(task_map, env):
    grid_size = task_map['grid_size']
    goal = tuple(task_map['goal_pos'])
    start = tuple(task_map['agent_start_pos'])

    # Возможные направления движения
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Очередь для BFS
    queue = [start]

    # Словарь для хранения посещённых клеток и родителей
    visited = {start: None}

    while queue:
        current = queue.pop(0)

        # Достигли цели
        if current == goal:
            break

        x, y = current

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (nx, ny) not in visited and is_passable(nx, ny, env, grid_size):
                queue.append((nx, ny))
                visited[(nx, ny)] = current

    # Восстановление пути
    path = []
    current = goal

    # Если цель недостижима
    if current not in visited:
        return []

    while current is not None:
        path.append(current)
        current = visited[current]

    path.reverse()

    return path