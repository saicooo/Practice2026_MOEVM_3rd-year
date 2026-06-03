#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 1000

typedef struct {
    int x, y;
} Point;

typedef struct {
    Point *data;
    int front, rear, size, capacity;
} Queue;

Queue* createQueue(int capacity) {
    Queue *q = (Queue*)malloc(sizeof(Queue));
    q->data = (Point*)malloc(capacity * sizeof(Point));
    q->front = 0;
    q->rear = -1;
    q->size = 0;
    q->capacity = capacity;
    return q;
}

void enqueue(Queue *q, Point p) {
    q->rear = (q->rear + 1) % q->capacity;
    q->data[q->rear] = p;
    q->size++;
}

Point dequeue(Queue *q) {
    Point p = q->data[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->size--;
    return p;
}

int isEmpty(Queue *q) {
    return q->size == 0;
}

void freeQueue(Queue *q) {
    free(q->data);
    free(q);
}

int main() {
    int N;
    scanf("%d", &N);
    
    int start_x, start_y, goal_x, goal_y;
    scanf("%d %d", &start_x, &start_y);
    scanf("%d %d", &goal_x, &goal_y);
    
    // Инициализация сетки (0 - проходимо, 1 - стена)
    char **grid = (char**)malloc(N * sizeof(char*));
    for (int i = 0; i < N; i++) {
        grid[i] = (char*)calloc(N, sizeof(char));
    }
    
    int W;
    scanf("%d", &W);
    for (int i = 0; i < W; i++) {
        int x, y;
        scanf("%d %d", &x, &y);
        grid[x][y] = 1;
    }
    
    // Массивы для BFS
    Point **parent = (Point**)malloc(N * sizeof(Point*));
    int **visited = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        parent[i] = (Point*)malloc(N * sizeof(Point));
        visited[i] = (int*)calloc(N, sizeof(int));
        for (int j = 0; j < N; j++) {
            parent[i][j].x = -1;
            parent[i][j].y = -1;
        }
    }
    
    // Направления движения: вверх, вниз, влево, вправо
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    Queue *q = createQueue(N * N);
    Point start = {start_x, start_y};
    enqueue(q, start);
    visited[start_x][start_y] = 1;
    
    int found = 0;
    
    while (!isEmpty(q) && !found) {
        Point current = dequeue(q);
        
        for (int i = 0; i < 4; i++) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            
            if (nx >= 0 && nx < N && ny >= 0 && ny < N && 
                !visited[nx][ny] && grid[nx][ny] == 0) {
                visited[nx][ny] = 1;
                parent[nx][ny] = current;
                
                if (nx == goal_x && ny == goal_y) {
                    found = 1;
                    break;
                }
                
                Point next = {nx, ny};
                enqueue(q, next);
            }
        }
    }
    
    // Восстановление пути
    Point *path = (Point*)malloc(N * N * sizeof(Point));
    int path_len = 0;
    
    Point current = {goal_x, goal_y};
    path[path_len++] = current;
    
    while (current.x != start_x || current.y != start_y) {
        current = parent[current.x][current.y];
        path[path_len++] = current;
    }
    
    // Вывод пути в обратном порядке
    for (int i = path_len - 1; i >= 0; i--) {
        printf("%d %d\n", path[i].x, path[i].y);
    }
    
    // Освобождение памяти
    for (int i = 0; i < N; i++) {
        free(grid[i]);
        free(parent[i]);
        free(visited[i]);
    }
    free(grid);
    free(parent);
    free(visited);
    free(path);
    freeQueue(q);
    
    return 0;
}