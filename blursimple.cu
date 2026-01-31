# @title
%%file /tmp/newblur.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

#define DODEBUG 1

#ifdef DODEBUG
#define CHECK_ERR(err, a) {if (err!= cudaSuccess) { \
 printf("err here %s %d %s ", __FILE__,__LINE__,a ); \
 ; \
}}
#else
#define CHECK_ERR(err, a) {}
#endif

//onethread
__global__ void kernelblur(const float* r_inarr,const float* g_inarr,const float* b_inarr, float* r_ouarr, float* g_ouarr, float* b_ouarr,const int width,const int height){
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x >=width || y>=height) return;

  auto clump = [](int cur, int low, int hig)-> int{ //newid
      return (cur<low) ? low : ((cur>hig) ? hig : cur);
  };
  //int nx = (x + dx < 0) ? 0 : ((x + dx >= width) ? width - 1 : x  +  dx);
  //int ny = (y + dy < 0) ? 0 : ((y + dy >= height) ? height - 1 : y + dy);

  float sum_r = 0.0f;
  float sum_g = 0.0f;
  float sum_b = 0.0f;
  int count =0;//9 always

  for (int dy = -1; dy<=1; dy++){//3x3 offset gen
  for (int dx = -1; dx<=1; dx++){
      int tempidx = clump(x+dx,0,width -1);
      int tempidy = clump(y+dy,0,height-1);
      int tempind = tempidy*width + tempidx;

      sum_r += r_inarr[tempind];
      sum_g += g_inarr[tempind];
      sum_b += b_inarr[tempind];
      count++;
  }
  }

  int outind = y*width + x;
  r_ouarr[outind] = sum_r/count;
  g_ouarr[outind] = sum_g/count;
  b_ouarr[outind] = sum_b/count;
}

//all at the same time
void cpublur(const float* r_inarr,const float* g_inarr,const float* b_inarr, float* r_ouarr, float* g_ouarr, float* b_ouarr,const int width,const int height){


  auto clump = [](int cur, int low, int hig)-> int{ //newid
      return (cur<low) ? low : ((cur>hig) ? hig : cur);
  };
  //int nx = (x + dx < 0) ? 0 : ((x + dx >= width) ? width - 1 : x   + dx);
  //int ny = (y + dy < 0) ? 0 : ((y + dy >= height) ? height - 1 : y + dy);

  //if (x >=width || y>=height) return;
  for (int x = 0; x<width;  x++){//whichpixelredo
  for (int y = 0; y<height; y++){
  float sum_r = 0.0f;
  float sum_g = 0.0f;
  float sum_b = 0.0f;
  int count =0;//9 always

  for (int dy = -1; dy<=1; dy++){//3x3 offset gen
    for (int dx = -1; dx<=1; dx++){
      int tempidx = clump(x+dx,0,width  -1);
      int tempidy = clump(y+dy,0,height -1);
      int tempind = tempidy*width + tempidx;
      sum_r += r_inarr[tempind];
      sum_g += g_inarr[tempind];
      sum_b += b_inarr[tempind];
      count++;
      }
    }//onepixeldonesum

  int outind = y*width + x;
  r_ouarr[outind] = sum_r/count;
  g_ouarr[outind] = sum_g/count;
  b_ouarr[outind] = sum_b/count;
  }
  }//allpixelsdone
}//cpukernend

//=============================================================================================

int main() {
    const int WIDTH    = 1024;
    const int HEIGH    = 1024;
    const int N_PXL    = WIDTH * HEIGH;
    const size_t SIZEA = N_PXL * sizeof(float);

    float *h_r_in, *h_g_in, *h_b_in;//hostin
    float *h_r_ou, *h_g_ou, *h_b_ou;//hostout
    float *g_r_ou, *g_g_ou, *g_b_ou;//hostoutofgpu

    cudaMallocHost(&h_r_in, SIZEA);
    cudaMallocHost(&h_g_in, SIZEA);
    cudaMallocHost(&h_b_in, SIZEA);
    cudaMallocHost(&g_r_ou, SIZEA);
    cudaMallocHost(&g_g_ou, SIZEA);
    cudaMallocHost(&g_b_ou, SIZEA);
    h_r_ou = (float*)malloc(SIZEA);
    h_g_ou = (float*)malloc(SIZEA);
    h_b_ou = (float*)malloc(SIZEA);

    if(!h_r_in||!h_g_in||!h_b_in||
       !h_r_ou||!h_g_ou||!h_b_ou||
       !g_r_ou||!g_g_ou||!g_b_ou){
       printf("Hostallocdead");
       return -1;
       }

    float *d_r_in, *d_g_in, *d_b_in;//devicein
    float *d_r_ou, *d_g_ou, *d_b_ou;//deviceout
    cudaMalloc(&d_r_in, SIZEA);
    cudaMalloc(&d_g_in, SIZEA);
    cudaMalloc(&d_b_in, SIZEA);
    cudaMalloc(&d_r_ou, SIZEA);
    cudaMalloc(&d_g_ou, SIZEA);
    cudaMalloc(&d_b_ou, SIZEA);
    CHECK_ERR(cudaGetLastError(), "cudaMalloc device");

    srand(456789);
    for (int i = 0; i<N_PXL; i++){
        h_r_in[i] = (float)rand() / RAND_MAX * 255.0f;
        h_g_in[i] = (float)rand() / RAND_MAX * 255.0f;
        h_b_in[i] = (float)rand() / RAND_MAX * 255.0f;
    }

    auto htstar = std::chrono::high_resolution_clock::now();
    cpublur(h_r_in,h_g_in,h_b_in,h_r_ou,h_g_ou,h_b_ou,WIDTH,HEIGH);
    auto htstop = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(htstop - htstar).count() / 1000.0f;
    printf("CPU time: %.3f ms\n", cpu_ms);

    cudaEvent_t tstar;
    cudaEvent_t tstop;
    cudaEventCreate(&tstar);
    cudaEventCreate(&tstop);

    cudaMemcpy(d_r_in, h_r_in, SIZEA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, h_g_in, SIZEA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b_in, SIZEA, cudaMemcpyHostToDevice);
    CHECK_ERR(cudaGetLastError(), "h2d");

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1)/blockSize.x,
                  (HEIGH + blockSize.y - 1)/blockSize.y);

         cudaEventRecord(tstar);
    kernelblur<<<gridSize,blockSize>>>(d_r_in,d_g_in,d_b_in,d_r_ou,d_g_ou,d_b_ou,WIDTH,HEIGH);
    CHECK_ERR(cudaGetLastError(), "kernelblur");
    cudaMemcpy(g_r_ou, d_r_ou, SIZEA, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_g_ou, d_g_ou, SIZEA, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_b_ou, d_b_ou, SIZEA, cudaMemcpyDeviceToHost);
    CHECK_ERR(cudaGetLastError(), "d2h");
         cudaEventRecord(tstop);
    cudaEventSynchronize(tstop);
    float gtime = -1;
    cudaEventElapsedTime(&gtime, tstar, tstop);
    printf("GPU time: %.3f ms\n", gtime);

    // --- Validation ---
    const float TOL = 1e-3f; // tolerance for floating-point comparison
    bool valid = true;
    int mismatch_count = 0;
    const int max_report = 5; // report first few mismatches

    for (int i = 0; i < N_PXL; ++i) {
    float dr = fabsf(h_r_ou[i] - g_r_ou[i]);
    float dg = fabsf(h_g_ou[i] - g_g_ou[i]);
    float db = fabsf(h_b_ou[i] - g_b_ou[i]);
    if (dr > TOL || dg > TOL || db > TOL) {
    if (mismatch_count < max_report) {
    printf("Mismatch at pixel %d: CPU(%.3f,%.3f,%.3f) vs GPU(%.3f,%.3f,%.3f)\n",
    i, h_r_ou[i], h_g_ou[i], h_b_ou[i],
    g_r_ou[i], g_g_ou[i], g_b_ou[i]);    }
    mismatch_count++;
    valid = false;
    }
    }
    if (valid) { printf(" Validation PASSED: %.1e\n", TOL);}
    else { printf("Validation FAILED: %d / %d pixels differ beyond tolerance %.1e\n", mismatch_count, N_PXL, TOL);}

    cudaFreeHost(h_r_in); cudaFreeHost(h_g_in); cudaFreeHost(h_b_in);
    cudaFreeHost(g_r_ou); cudaFreeHost(g_g_ou); cudaFreeHost(g_b_ou);
    free(h_r_ou); free(h_g_ou); free(h_b_ou);

    cudaFree(d_r_in); cudaFree(d_g_in); cudaFree(d_b_in);
    cudaFree(d_r_ou); cudaFree(d_g_ou); cudaFree(d_b_ou);
    cudaEventDestroy(tstar);
    cudaEventDestroy(tstop);

    return 0;
}