# @title
%%file /tmp/t1v1.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <chrono>

// Параметры задачи
#define NX 256           // Количество узлов по x
#define NY 256           // Количество узлов по y
#define DX (1.0/(NX-1)) // Шаг по пространству
#define DY (1.0/(NY-1))
#define C 1.0           // Коэффициент теплопроводности

// Автоматический расчет устойчивого шага по времени
#define DT (0.25 * fmin(DX*DX, DY*DY) / C) // Условие Куранта

double** allocate_2d_array(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double*)calloc(cols, sizeof(double)); // Используем calloc для инициализации нулями
    }
    return arr;
}

void free_2d_array(double** arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Инициализация начальных и граничных условий
void initialize(double** u) {
    // Начальное условие: u(x,y,0) = 1.0
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            u[i][j] = 1.0;
        }
    }

    // Граничные условия: u = 2.0 на всех границах
    for (int i = 0; i < NX; i++) {
        u[i][0] = 2.0;
        u[i][NY-1] = 2.0;
    }

    for (int j = 0; j < NY; j++) {
        u[0][j] = 2.0;
        u[NX-1][j] = 2.0;
    }
}

// Один шаг по времени методом крест
void time_step(double** u, double** u_new) {
    double dx2 = DX * DX;
    double dy2 = DY * DY;

    // Копируем граничные условия
    for (int i = 0; i < NX; i++) {
        u_new[i][0] = u[i][0];
        u_new[i][NY-1] = u[i][NY-1];
    }
    for (int j = 0; j < NY; j++) {
        u_new[0][j] = u[0][j];
        u_new[NX-1][j] = u[NX-1][j];
    }

    // Вычисление новых значений во внутренних точках
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            double laplacian = (u[i-1][j] - 2*u[i][j] + u[i+1][j]) / dx2 +
                              (u[i][j-1] - 2*u[i][j] + u[i][j+1]) / dy2;

            u_new[i][j] = u[i][j] + DT * C * laplacian;

            // Проверка на численную устойчивость
            if (!std::isfinite(u_new[i][j])) {
                printf("Numerical instability at (%d, %d)! Value = %f\n", i, j, u_new[i][j]);
                return;
            }
        }
    }
}

// Вычисление максимального изменения за шаг (для мониторинга)
double max_change(double** u, double** u_new) {
    double max_diff = 0.0;
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            double diff = fabs(u_new[i][j] - u[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

void save_to_file(double** u, const char* filename, double t) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    fprintf(file, "# Time: %.6f\n", t);
    fprintf(file, "# X Y U\n");

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(file, "%.6f %.6f %.6f\n", i*DX, j*DY, u[i][j]);
        }
    }

    fclose(file);
    printf("Saved to %s\n", filename);
}

int main(int argc, char* argv[]) {
    double target_time = 0.1;
    if (target_time <= 0) {
        printf("Error: target_time must be positive\n");
        return 1;
    }

    printf("Solving heat equation until t = %.6f\n", target_time);
    printf("Grid: %dx%d, dx=%.6f, dy=%.6f\n", NX, NY, DX, DY);
    printf("Automatic time step: dt=%.8f (for stability)\n", DT);
    printf("Courant number: %.6f\n", C * DT * (1.0/(DX*DX) + 1.0/(DY*DY)));

    double** u_current = allocate_2d_array(NX, NY);
    double** u_next = allocate_2d_array(NX, NY);
    initialize(u_current);
    initialize(u_next);

    double t = 0.0;
    int step = 0;
    printf("Step 0: t = %.6f\n", t);
    auto start_computation = std::chrono::high_resolution_clock::now();

    // Основной цикл по времени
    while (t < target_time) {
        // Если осталось меньше чем DT, уменьшаем шаг
        double actual_dt = DT;
        if (t + DT > target_time) {
            actual_dt = target_time - t;
        }

        time_step(u_current, u_next);
        std::swap(u_current, u_next);

        t += actual_dt;
        step++;

        // Вывод прогресса каждые 1000 шагов
        if (step % 1000 == 0) {
            double max_diff = max_change(u_next, u_current);
            printf("Step %d: t = %.6f, max change = %.8f\n", step, t, max_diff);
        }

        // Аварийный выход при неустойчивости
        if (step > 1000000) { // Защита от бесконечного цикла
            printf("Too many steps, possible instability!\n");
            break;
        }
    }

    auto end_computation = std::chrono::high_resolution_clock::now();

    printf("Final: step %d, t = %.6f\n", step, t);

    save_to_file(u_current, "solution_final.txt", t);

    // СИММЕТРИЧНЫЙ вывод среза для проверки (ИСПРАВЛЕННЫЙ)
    printf("\nSymmetric slice at y = %d (middle):\n", NY/2);
    printf("Left part (from center to left boundary):\n");

    // Используем безопасные границы
    int center = NX / 2;
    int max_offset = center; // максимальный offset до левой границы

    for (int offset = 0; offset <= max_offset; offset += NX/16) {
        int left_x = center - offset;
        int right_x = center + offset;

        // Проверяем границы для безопасности
        if (left_x >= 0 && left_x < NX && right_x >= 0 && right_x < NX) {
            printf("  u[%3d][%3d] = %8.6f  |  u[%3d][%3d] = %8.6f\n",
                   left_x, NY/2, u_current[left_x][NY/2],
                   right_x, NY/2, u_current[right_x][NY/2]);
        }
    }

    // Проверка симметрии более детально (только для безопасных offset)
    printf("\nDetailed symmetry check around center:\n");
    for (int offset = 1; offset <= 5; offset++) {
        int left_x = center - offset;
        int right_x = center + offset;

        if (left_x >= 0 && right_x < NX) {
            double left_val = u_current[left_x][NY/2];
            double right_val = u_current[right_x][NY/2];
            double symmetry_error = fabs(left_val - right_val);
            printf("  Offset %d: u[%d]=%.6f, u[%d]=%.6f, error=%.8f\n",
                   offset, left_x, left_val, right_x, right_val, symmetry_error);
        }
    }

    printf("\nBoundary check:\n");
    printf("Top-left: u[0][%d] = %.6f (should be 2.0)\n", NY-1, u_current[0][NY-1]);
    printf("Center: u[%d][%d] = %.6f\n", NX/2, NY/2, u_current[NX/2][NY/2]);
    printf("Bottom-right: u[%d][0] = %.6f (should be 2.0)\n", NX-1, u_current[NX-1][0]);

    auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end_computation - start_computation);
    printf("\n=== Performance Results ===\n");
    printf("Computation time: %.3f ms\n", computation_time.count() / 1000.0);
    printf("Time per step: %.3f microseconds\n", computation_time.count() / (double)step);

    free_2d_array(u_current, NX);
    free_2d_array(u_next, NX);

    return 0;
}