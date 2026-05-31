from collections import deque


def solution(task_map):
    n = task_map['grid_size']
    start = task_map['agent_start_pos']  # уже [x, y]
    walls = task_map['walls']  # уже [[x, y], ...]
    keys = task_map['keys']  # уже [[x, y], ...]

    # Убираем дубликаты ключей
    seen = set()
    unique_keys = []
    for k in keys:
        p = tuple(k)  # (x, y)
        if p not in seen:
            seen.add(p)
            unique_keys.append(k)

    if not unique_keys:
        return [start]

    # Для внутренних вычислений используем (row, col) = (y, x)
    def to_row_col(xy):
        return (xy[1], xy[0])  # (y, x)
    
    def to_xy(row_col):
        return [row_col[1], row_col[0]]  # [x, y]

    start_rc = to_row_col(start)
    walls_rc = set(to_row_col(w) for w in walls)
    keys_rc = [to_row_col(k) for k in unique_keys]

    # Быстрое индексирование клетки -> int
    def idx(r, c):
        return r * n + c

    def pos(i):
        r, c = divmod(i, n)
        return (r, c)

    # Предвычислим соседей для всех клеток
    total = n * n
    blocked = [False] * total
    for (r, c) in walls_rc:
        blocked[idx(r, c)] = True

    neighbors = [[] for _ in range(total)]
    for r in range(n):
        for c in range(n):
            if blocked[idx(r, c)]:
                continue
            i = idx(r, c)
            if r > 0 and not blocked[idx(r - 1, c)]:
                neighbors[i].append(idx(r - 1, c))
            if r + 1 < n and not blocked[idx(r + 1, c)]:
                neighbors[i].append(idx(r + 1, c))
            if c > 0 and not blocked[idx(r, c - 1)]:
                neighbors[i].append(idx(r, c - 1))
            if c + 1 < n and not blocked[idx(r, c + 1)]:
                neighbors[i].append(idx(r, c + 1))

    # BFS от одной точки: dist + parent
    def bfs(src_i):
        dist = [-1] * total
        parent = [-1] * total
        q = deque([src_i])
        dist[src_i] = 0
        while q:
            cur = q.popleft()
            for nx in neighbors[cur]:
                if dist[nx] == -1:
                    dist[nx] = dist[cur] + 1
                    parent[nx] = cur
                    q.append(nx)
        return dist, parent

    def restore(parent, src_i, dst_i):
        if src_i == dst_i:
            return [pos(src_i)]
        if parent[dst_i] == -1:
            return None
        path = []
        p = dst_i
        while p != -1:
            path.append(pos(p))
            if p == src_i:
                break
            p = parent[p]
        path.reverse()
        if path and path[0] == pos(src_i):
            return path
        return None

    # Узлы интереса
    nodes_rc = [start_rc] + keys_rc
    node_idx = [idx(r, c) for (r, c) in nodes_rc]
    m = len(nodes_rc)

    # BFS из каждой интересующей точки
    dists = []
    parents = []
    for s_i in node_idx:
        d, p = bfs(s_i)
        dists.append(d)
        parents.append(p)

    INF = 10**9
    dist_mat = [[INF] * m for _ in range(m)]
    for i in range(m):
        di = dists[i]
        for j in range(m):
            v = di[node_idx[j]]
            if v != -1:
                dist_mat[i][j] = v

    # Проверка достижимости
    for j in range(1, m):
        if dist_mat[0][j] >= INF:
            return [start]

    K = m - 1

    # Если ключей слишком много — эвристика
    if K > 15:
        unvisited = set(range(1, m))
        order = []
        cur = 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: dist_mat[cur][j])
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
    else:
        # Точный DP по битмаске
        FULL = (1 << K) - 1
        dp = [[INF] * m for _ in range(1 << K)]
        prev = [[None] * m for _ in range(1 << K)]
        dp[0][0] = 0

        for mask in range(1 << K):
            row = dp[mask]
            for last in range(m):
                curd = row[last]
                if curd >= INF:
                    continue
                for k in range(K):
                    bit = 1 << k
                    if mask & bit:
                        continue
                    nxt = k + 1
                    nd = curd + dist_mat[last][nxt]
                    nmask = mask | bit
                    if nd < dp[nmask][nxt]:
                        dp[nmask][nxt] = nd
                        prev[nmask][nxt] = (mask, last)

        best_last = min(range(1, m), key=lambda j: dp[FULL][j])
        if dp[FULL][best_last] >= INF:
            return [start]

        order = []
        mask = FULL
        last = best_last
        while last != 0:
            order.append(last)
            pm = prev[mask][last]
            if pm is None:
                break
            mask, last = pm
        order.reverse()

    # Построение полного маршрута
    route = [start]  # уже в формате [x, y]
    cur_idx = 0
    cur_pos_i = node_idx[0]
    cur_pos_rc = nodes_rc[0]

    for node_i in order:
        target_rc = nodes_rc[node_i]
        target_i = node_idx[node_i]
        seg = restore(parents[cur_idx], cur_pos_i, target_i)
        if seg is None:
            return [start]
        # seg содержит кортежи (row, col), конвертируем в [x, y]
        for pos_rc in seg[1:]:
            route.append(to_xy(pos_rc))
        cur_idx = node_i
        cur_pos_i = target_i

    return route