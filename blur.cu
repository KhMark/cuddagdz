# @title
%%file /tmp/t2v1.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

#define _DEBUGY 1

#ifdef _DEBUGY
#define CHECK_ERR(err, a) { if (err != cudaSuccess) { \
    printf("%s(%d): %s\n", __FILE__, __LINE__, a); \
    } \
}
#else
#define CHECK_ERR(err, a) {}
#endif

__global__ void blurKernelSoA(
    const float* __restrict__ r_in,
    const float* __restrict__ g_in,
    const float* __restrict__ b_in,
    float* __restrict__ r_out,
    float* __restrict__ g_out,
    float* __restrict__ b_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    auto clamp = [](int v, int lo, int hi) -> int {
        return (v < lo) ? lo : (v > hi ? hi : v);
    };

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            int idx = ny * width + nx;

            sum_r += r_in[idx];
            sum_g += g_in[idx];
            sum_b += b_in[idx];
            count++;
        }
    }

    int out_idx = y * width + x;
    r_out[out_idx] = sum_r / count;
    g_out[out_idx] = sum_g / count;
    b_out[out_idx] = sum_b / count;
}

void cpuBlurSoA(
    const float* r_in, const float* g_in, const float* b_in,
    float* r_out, float* g_out, float* b_out,
    int width, int height)
{
    auto clamp = [](int v, int lo, int hi) -> int {
        return (v < lo) ? lo : (v > hi ? hi : v);
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = clamp(x + dx, 0, width - 1);
                    int ny = clamp(y + dy, 0, height - 1);
                    int idx = ny * width + nx;

                    sum_r += r_in[idx];
                    sum_g += g_in[idx];
                    sum_b += b_in[idx];
                    count++;
                }
            }

            int out_idx = y * width + x;
            r_out[out_idx] = sum_r / count;
            g_out[out_idx] = sum_g / count;
            b_out[out_idx] = sum_b / count;
        }
    }
}

//=============================================================================================

int main() {
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int NUM_PIXELS = WIDTH * HEIGHT;
    size_t size = NUM_PIXELS * sizeof(float);

    printf("SoA Blur: Image size %dx%d (%d pixels)\n", WIDTH, HEIGHT, NUM_PIXELS);

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    float *h_r_in, *h_g_in, *h_b_in;
    float *h_r_out_gpu, *h_g_out_gpu, *h_b_out_gpu;
    cudaMallocHost(&h_r_in, size);
    cudaMallocHost(&h_g_in, size);
    cudaMallocHost(&h_b_in, size);
    cudaMallocHost(&h_r_out_gpu, size);
    cudaMallocHost(&h_g_out_gpu, size);
    cudaMallocHost(&h_b_out_gpu, size);

    float *h_r_out_cpu = (float*)malloc(size);
    float *h_g_out_cpu = (float*)malloc(size);
    float *h_b_out_cpu = (float*)malloc(size);

    if (!h_r_in || !h_g_in || !h_b_in ||
        !h_r_out_gpu || !h_g_out_gpu || !h_b_out_gpu ||
        !h_r_out_cpu || !h_g_out_cpu || !h_b_out_cpu) {
        printf("Host alloc failed!\n");
        return -1;
    }

    float *d_r_in, *d_g_in, *d_b_in;
    float *d_r_out, *d_g_out, *d_b_out;
    cudaMalloc(&d_r_in, size);
    cudaMalloc(&d_g_in, size);
    cudaMalloc(&d_b_in, size);
    cudaMalloc(&d_r_out, size);
    cudaMalloc(&d_g_out, size);
    cudaMalloc(&d_b_out, size);
    CHECK_ERR(cudaGetLastError(), "cudaMalloc device");

    srand(42);
    for (int i = 0; i < NUM_PIXELS; i++) {
        h_r_in[i] = (float)rand() / RAND_MAX * 255.0f;
        h_g_in[i] = (float)rand() / RAND_MAX * 255.0f;
        h_b_in[i] = (float)rand() / RAND_MAX * 255.0f;
    }

    auto tcpu_start = std::chrono::high_resolution_clock::now();
    cpuBlurSoA(h_r_in, h_g_in, h_b_in,
               h_r_out_cpu, h_g_out_cpu, h_b_out_cpu,
               WIDTH, HEIGHT);
    auto tcpu_stop = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(tcpu_stop - tcpu_start).count() / 1000.0f;
    printf("CPU time: %.3f ms\n", cpu_ms);

    cudaMemcpy(d_r_in, h_r_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, h_g_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b_in, size, cudaMemcpyHostToDevice);
    CHECK_ERR(cudaGetLastError(), "H2D input");

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(t_start);
    blurKernelSoA<<<gridSize, blockSize>>>(
        d_r_in, d_g_in, d_b_in,
        d_r_out, d_g_out, d_b_out,
        WIDTH, HEIGHT);
    CHECK_ERR(cudaGetLastError(), "blurKernelSoA launch");

    cudaMemcpy(h_r_out_gpu, d_r_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_out_gpu, d_g_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out_gpu, d_b_out, size, cudaMemcpyDeviceToHost);
    CHECK_ERR(cudaGetLastError(), "D2H output");

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, t_start, t_stop);
    printf("GPU time: %.3f ms\n", gpu_ms);

    const float tolerance = 1e-4f;
    int errors = 0;
    float max_err = 0.0f;
    const int sample_count = 1000;

    for (int i = 0; i < sample_count; i++) {
        int idx = rand() % NUM_PIXELS;
        float dr = fabsf(h_r_out_gpu[idx] - h_r_out_cpu[idx]);
        float dg = fabsf(h_g_out_gpu[idx] - h_g_out_cpu[idx]);
        float db = fabsf(h_b_out_gpu[idx] - h_b_out_cpu[idx]);
        float err = fmaxf(fmaxf(dr, dg), db);

        if (err > tolerance) {
            errors++;
            if (errors <= 3) {
                printf("Mismatch at %d: GPU(%.3f,%.3f,%.3f) CPU(%.3f,%.3f,%.3f) diff=%.6f\n",
                       idx,
                       h_r_out_gpu[idx], h_g_out_gpu[idx], h_b_out_gpu[idx],
                       h_r_out_cpu[idx], h_g_out_cpu[idx], h_b_out_cpu[idx],
                       err);
            }
        }
        if (err > max_err) max_err = err;
    }

    printf("\nSoA Validation: %d errors in %d samples, max error = %.6f\n",
           errors, sample_count, max_err);

    cudaFreeHost(h_r_in); cudaFreeHost(h_g_in); cudaFreeHost(h_b_in);
    cudaFreeHost(h_r_out_gpu); cudaFreeHost(h_g_out_gpu); cudaFreeHost(h_b_out_gpu);
    free(h_r_out_cpu); free(h_g_out_cpu); free(h_b_out_cpu);
    cudaFree(d_r_in); cudaFree(d_g_in); cudaFree(d_b_in);
    cudaFree(d_r_out); cudaFree(d_g_out); cudaFree(d_b_out);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);

    return 0;

}
