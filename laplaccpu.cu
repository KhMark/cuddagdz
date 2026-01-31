# @title
%%file /tmp/t1v1.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <chrono>

// Параметры задачи
//#define NX 256           // Количество узлов по x
//#define NY 256           // Количество узлов по y
#define NX 128           // Количество узлов по x
#define NY 128           // Количество узлов по y
#define DX (1.0/(NX-1)) // Шаг по пространству
#define DY (1.0/(NY-1))
#define MAX_ITER 50000   // Максимальное количество итераций
#define TOLERANCE 1e-5   // Точность решения

double** allocate_2d_array(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double*)calloc(cols, sizeof(double));
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
    // Внутренняя область инициализируется нулями
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            u[i][j] = 0.0;
        }
    }

    // Граничные условия из новой задачи
    for (int i = 0; i < NX; i++) {
        u[i][0] = exp(1.0 - i*DX);    // u(x,0) = e^(1-x)
        u[i][NY-1] = 1.0;             // u(x,1) = 1.0
    }

    for (int j = 0; j < NY; j++) {
        u[0][j] = exp(1.0 - j*DY);    // u(0,y) = e^(1-y)
        u[NX-1][j] = 1.0;             // u(1,y) = 1.0
    }
}

// Простой метод Якоби для уравнения Лапласа
void jacobi_step(double** u, double** u_new) {
    double dx2 = DX * DX;
    double dy2 = DY * DY;
    double factor = 0.5 / (1.0/dx2 + 1.0/dy2);

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
            u_new[i][j] = factor * (
                (u[i-1][j] + u[i+1][j]) / dx2 +
                (u[i][j-1] + u[i][j+1]) / dy2
            );
        }
    }
}

// Вычисление максимального изменения за шаг
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

// Вычисление невязки (норма C)//------------------------------------------------------------------------
double compute_residual(double** u) {
    double max_residual = 0.0;
    double dx2 = DX * DX;
    double dy2 = DY * DY;

    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            double residual = fabs(
                (u[i-1][j] - 2*u[i][j] + u[i+1][j]) / dx2 +
                (u[i][j-1] - 2*u[i][j] + u[i][j+1]) / dy2
            );
            if (residual > max_residual) {
                max_residual = residual;
            }
        }
    }
    return max_residual;
}

// Вычисление нормы L1
double compute_l1_norm(double** u) {
    double sum = 0.0;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            sum += fabs(u[i][j]);
        }
    }
    return sum * DX * DY;
}

void save_to_file(double** u, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

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
    printf("Solving Laplace equation: d²u/dx² + d²u/dy² = 0\n");
    printf("Grid: %dx%d, dx=%.6f, dy=%.6f\n", NX, NY, DX, DY);
    printf("Tolerance: %.2e, Max iterations: %d\n", TOLERANCE, MAX_ITER);

    double** u_current = allocate_2d_array(NX, NY);
    double** u_next = allocate_2d_array(NX, NY);
    initialize(u_current);
    initialize(u_next);

    auto start_computation = std::chrono::high_resolution_clock::now();

    // Основной итерационный цикл
    int iter = 0;
    double max_diff = 0.0;
    double residual = 0.0;

    printf("\nStarting iterations...\n");
    printf("Iter  Residual(C-norm)    Max-Change      L1-Norm\n");
    printf("--------------------------------------------------\n");

    for (iter = 1; iter <= MAX_ITER; iter++) {
        jacobi_step(u_current, u_next);

        max_diff = max_change(u_current, u_next);
        residual = compute_residual(u_next);
        double l1_norm = compute_l1_norm(u_next);

        std::swap(u_current, u_next);

        // Вывод прогресса каждые 1000 итераций
        if (iter % 1000 == 0) {
            printf("%5d  %12.6e  %12.6e  %12.6e\n",
                   iter, residual, max_diff, l1_norm);
        }

        // Критерий остановки
        if (residual < TOLERANCE && max_diff < TOLERANCE) {
            printf("%5d  %12.6e  %12.6e  %12.6e\n",
                   iter, residual, max_diff, l1_norm);
            printf("Converged!\n");
            break;
        }
    }

    auto end_computation = std::chrono::high_resolution_clock::now();

    if (iter > MAX_ITER) {
        printf("Reached maximum iterations without convergence\n");
        printf("Current residual: %.6e (target: %.1e)\n", residual, TOLERANCE);
    }

    printf("\n=== Final Results ===\n");
    printf("Iterations: %d\n", iter);
    printf("Final residual (C-norm): %.6e\n", residual);
    printf("Final max change: %.6e\n", max_diff);
    printf("Final L1 norm: %.6e\n", compute_l1_norm(u_current));

    save_to_file(u_current, "solution_laplace.txt");

    // Анализ решения
    printf("\n=== Solution Analysis ===\n");

    // Проверка граничных условий
    printf("Boundary conditions check:\n");
    printf("u(0,0) = %.6f (should be e = %.6f)\n", u_current[0][0], exp(1.0));
    printf("u(1,0) = %.6f (should be 1.0)\n", u_current[NX-1][0]);
    printf("u(0,1) = %.6f (should be 1.0)\n", u_current[0][NY-1]);
    printf("u(1,1) = %.6f (should be 1.0)\n", u_current[NX-1][NY-1]);

    // Проверка внутренних точек
    printf("\nInternal points:\n");
    printf("u(0.5, 0.5) = %.6f\n", u_current[NX/2][NY/2]);
    printf("u(0.25, 0.25) = %.6f\n", u_current[NX/4][NY/4]);
    printf("u(0.75, 0.75) = %.6f\n", u_current[3*NX/4][3*NY/4]);

    auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end_computation - start_computation);
    printf("\n=== Performance Results ===\n");
    printf("Total computation time: %.3f ms\n", computation_time.count() / 1000.0);
    printf("Time per iteration: %.3f microseconds\n", computation_time.count() / (double)iter);
    printf("Iterations per second: %.0f\n", iter / (computation_time.count() / 1000000.0));

    free_2d_array(u_current, NX);
    free_2d_array(u_next, NX);

    return 0;
}