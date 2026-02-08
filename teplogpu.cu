# @title
%%file /tmp/t2v1.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>

#define NX 256
#define NY 256
#define DX (1.0f/(NX-1))
#define DY (1.0f/(NY-1))
#define C 1.0f
#define DT (0.25f * fminf(DX*DX, DY*DY) / C)

#define BLOCK_SIZE 16

#define CHECK_CUDA_ERROR(err) \
if (err != cudaSuccess) { \
     printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
     exit(1); \
    }


__global__ void time_step_kernel(double* u, double* u_new, int width, int height,
                                double dx2, double dy2, double dt_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;

    if (i == 0 || i == width-1 || j == 0 || j == height-1) {
        u_new[idx] = u[idx];
        return;
    }

    double laplacian = (u[idx - 1] - 2*u[idx] + u[idx + 1]) / dx2 +
                      (u[idx - width] - 2*u[idx] + u[idx + width]) / dy2;

    u_new[idx] = u[idx] + dt_c * C * laplacian;
}

__global__ void initialize_kernel(double* u, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;
    u[idx] = 1.0;

    if (i == 0 || i == width-1 || j == 0 || j == height-1) {
        u[idx] = 2.0;
    }
}

double compute_max_change_cpu(double* u, double* u_new, int width, int height) {
    double max_diff = 0.0;
    for (int j = 1; j < height-1; j++) {
        for (int i = 1; i < width-1; i++) {
            int idx = j * width + i;
            double diff = fabs(u_new[idx] - u[idx]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

double* allocate_gpu_memory(int size) {
    double* ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, size * sizeof(double)));
    return ptr;
}

void free_gpu_memory(double* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

void copy_between_cpu_gpu(double* dst, double* src, int size, bool to_gpu) {
    cudaMemcpyKind kind = to_gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size * sizeof(double), kind));
}

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

void save_to_file(double* u, const char* filename, double t, int width, int height) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    fprintf(file, "# Time: %.6f\n", t);
    fprintf(file, "# X Y U\n");

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int idx = j * width + i;
            fprintf(file, "%.6f %.6f %.6f\n", i*DX, j*DY, u[idx]);
        }
    }

    fclose(file);
    printf("Saved to %s\n", filename);
}

int main() {
    double target_time = 0.1;

    printf("Solving heat equation with CUDA (SYMMETRIC OUTPUT) until t = %.6f\n", target_time);
    printf("Grid: %dx%d, dx=%.6f, dy=%.6f\n", NX, NY, DX, DY);
    printf("Time step: dt=%.8f\n", DT);
    printf("Block size: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);

    double* h_u = (double*)malloc(NX * NY * sizeof(double));
    double* h_u_new = (double*)malloc(NX * NY * sizeof(double));
    double* d_u = allocate_gpu_memory(NX * NY);
    double* d_u_new = allocate_gpu_memory(NX * NY);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid size: %dx%d blocks\n", gridSize.x, gridSize.y);

    for (int i = 0; i < NX * NY; i++) {
        h_u[i] = 1.0;
    }
    for (int i = 0; i < NX; i++) {
        h_u[i] = 2.0;
        h_u[(NY-1)*NX + i] = 2.0;
    }
    for (int j = 0; j < NY; j++) {
        h_u[j*NX] = 2.0;
        h_u[j*NX + (NX-1)] = 2.0;
    }

    copy_between_cpu_gpu(d_u, h_u, NX * NY, true);
    copy_between_cpu_gpu(d_u_new, h_u, NX * NY, true);

    double t = 0.0;
    int step = 0;

    printf("Step 0: t = %.6f\n", t);

    auto start_computation = std::chrono::high_resolution_clock::now();
    while (t < target_time) {
        double actual_dt = DT;
        if (t + DT > target_time) {
            actual_dt = target_time - t;
        }

        time_step_kernel<<<gridSize, blockSize>>>(d_u, d_u_new, NX, NY, DX*DX, DY*DY, actual_dt);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::swap(d_u, d_u_new);
        t += actual_dt;
        step++;

        if (step % 1000 == 0) {
            copy_between_cpu_gpu(h_u, d_u, NX * NY, false);
            copy_between_cpu_gpu(h_u_new, d_u_new, NX * NY, false);

            double max_diff = compute_max_change_cpu(h_u, h_u_new, NX, NY);
            printf("Step %d: t = %.6f, max change = %.8f\n", step, t, max_diff);
        }

        if (step > 1000000) {
            printf("Too many steps, possible instability!\n");
            break;
        }
    }

    auto end_computation = std::chrono::high_resolution_clock::now();

    printf("Final: step %d, t = %.6f\n", step, t);

    copy_between_cpu_gpu(h_u, d_u, NX * NY, false);
    save_to_file(h_u, "solution_final_cuda_symmetric.txt", t, NX, NY);

    printf("\nSymmetric slice at y = %d (middle):\n", NY/2);
    printf("Left part (from center to left boundary):\n");
    for (int offset = 0; offset <= NX/2; offset += NX/16) {
        int left_idx = (NY/2) * NX + (NX/2 - offset);
        int right_idx = (NY/2) * NX + (NX/2 + offset);
        printf("  u[%3d][%3d] = %8.6f  |  u[%3d][%3d] = %8.6f\n",
               NX/2 - offset, NY/2, h_u[left_idx],
               NX/2 + offset, NY/2, h_u[right_idx]);
    }

    printf("\nDetailed symmetry check around center:\n");
    int center = NX / 2;
    for (int offset = 1; offset <= 5; offset++) {
        int left_idx = (NY/2) * NX + (center - offset);
        int right_idx = (NY/2) * NX + (center + offset);
        double left_val = h_u[left_idx];
        double right_val = h_u[right_idx];
        double symmetry_error = fabs(left_val - right_val);
        printf("  Offset %d: u[%d]=%.6f, u[%d]=%.6f, error=%.8f\n",
               offset, center-offset, left_val, center+offset, right_val, symmetry_error);
    }

    printf("\nBoundary check:\n");
    printf("Top-left corner:     u[0][%d] = %.6f (should be 2.0)\n", NY-1, h_u[(NY-1)*NX + 0]);
    printf("Center:              u[%d][%d] = %.6f\n", NX/2, NY/2, h_u[(NY/2)*NX + NX/2]);
    printf("Bottom-right corner: u[%d][0] = %.6f (should be 2.0)\n", NX-1, h_u[0*NX + (NX-1)]);

    auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end_computation - start_computation);
    printf("\n=== CUDA Performance Results ===\n");
    printf("Computation time: %.3f ms\n", computation_time.count() / 1000.0);
    printf("Time per step: %.3f microseconds\n", computation_time.count() / (double)step);

    free(h_u);
    free(h_u_new);
    free_gpu_memory(d_u);
    free_gpu_memory(d_u_new);

    return 0;

}
