#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>

#define NX 25
#define NY 25
#define DX (1.0/(NX-1))
#define DY (1.0/(NY-1))
#define DX2 (DX*DX)
#define DY2 (DY*DY)
#define C 1.0    
#define DT (0.25 * fmin(DX*DX, DY*DY) / C) 



#define BLOCK_SIZE 16

#define CHECK_ERR(err,a) {\
    if (err!=cudaSuccess) {\
        printf("CUDA err at %s %d: %s (%s)\n",__FILE__,__LINE__,cudaGetErrorString(err),a); \
        exit(EXIT_FAILURE); \
    }\
}

double** allocate_2d_array(int rows, int cols){
    double** a = (double**) malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++){
        a[i] = (double*) calloc(cols, sizeof(double));//calloc -> 0,..,0
    }
    return a;
}

void free_2_array(double** a, int rows){
    for(int i=0; i<rows;i++){
        free(a[i]);
    }
    free(a);
}

void init_border(double** u){
    //u(X,0) = 1.0 | t=0
    for (int i =0; i<NX; i++){
    for (int ii=0;ii<NY;ii++){
        u[i][ii] = 1.0;
    }
    }
    //u(x,G,T) = 2.0
    for (int i=0; i<NX; i++){
        u[i][0]    =2.0;
        u[i][NY-1] =2.0;
    }
    //u(G,y,T) = 2.0
    for (int i=0; i<NY; i++){
        u[0][i]    =2.0;
        u[NX-1][i] =2.0;
    }
}

void init_border_1d(double* u){
    //u(X,0) = 1.0 | t=0
    for (int i =0; i<NX*NY; i++){
        u[i] = 1.0;
    }
    //u(x,G,T) = 2.0
    for (int i=0; i<NX; i++){
        u[i]    =2.0;
        u[NX*(NY-1) + i]    =2.0;
    }
    //u(G,y,T) = 2.0
    for (int i=0; i<NY; i++){
        u[NX*i]    =2.0;
        u[NX*i+  NX-1]    =2.0;
    }
}

void cross_tstep(double** u, double** u_new,double dt){
    //u(x,G,t) = 2.0
    for (int i=0; i<NX; i++){
        u_new[i][0]    =u[i][0]   ;
        u_new[i][NY-1] =u[i][NY-1];
    }
    //u(G,y,t) = 2.0
    for (int i=0; i<NX; i++){
        u_new[0][i]    =u[0][i]   ;
        u_new[NX-1][i] =u[NX-1][i];
    }
    //inner
    for (int i=1; i<NX-1;i++){
    for (int j=1; j<NY-1;j++){
        double laplac = (u[i-1][j] - 2*u[i][j]+ u[i+1][j])/DX2+
                        (u[i][j-1] - 2*u[i][j]+ u[i][j+1])/DY2;
        u_new[i][j] = u[i][j] + dt*C*laplac;

        if (!std::isfinite(u_new[i][j])) { // Проверка на численную устойчивость
            printf("Numerical instability at (%d, %d)! Value = %f\n", i, j, u_new[i][j]);
            return; }
    }
    }
}

//

__global__ void heat_step_kernel(const double*  u_curr, double* u_next,  double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < NX-1 && j >= 1 && j < NY-1) {
        const int idx = j * NX + i;
        
        double laplac = (u_curr[j * NX + (i-1)] - 2.0 * u_curr[idx] + u_curr[j * NX + (i+1)]) * (1.0/DX2) +
                        (u_curr[(j-1) * NX + i] - 2.0 * u_curr[idx] + u_curr[(j+1) * NX + i]) * (1.0/DY2);
        
        u_next[idx] = u_curr[idx] + dt * C * laplac;
    }
}




void print_matrix_2d(double** u) {
    int start_i = NX/2 - 4;
    int start_j = NY/2 - 4;
    //for (int j = start_j; j < start_j+7; j++) {
    //    for (int i = start_i; i < start_i+7; i++) {
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            printf("%.4f ", u[i][j]);
        }
        printf("\n");
    }
}

void print_matrix_1d(double* u) {
    int start_i = NX/2 - 4;
    int start_j = NY/2 - 4;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            printf("%.4f ", u[j*NX + i]);
        }
        printf("\n");
    }
}




//


int main(int argc, char* argv[]){
    double time_cur = 0.0;
    double time_end = 0.1;
    int step_cur = 0;
    double hosttime_cur = 0.0;
    int hoststep_cur = 0;

    double** u1=allocate_2d_array(NX,NY);
    double** u2=allocate_2d_array(NX,NY);
    init_border(u1);
    init_border(u2);

    //double* hostu1= (double*) malloc (NX*NY*sizeof(double));
    //double* hostu2= (double*) malloc (NX*NY*sizeof(double));
    //double* deviceu1= ;
    double* hostu1;
    double* hostu2;
    double* devsu1;
    double* devsu2;
    cudaMallocHost(&hostu1, (NX*NY*sizeof(double)) );
    cudaMallocHost(&hostu2, (NX*NY*sizeof(double)) );
    CHECK_ERR(cudaGetLastError(),"hostmalloc");
    cudaMalloc    (&devsu1, (NX*NY*sizeof(double)) );
    cudaMalloc    (&devsu2, (NX*NY*sizeof(double)) );
    CHECK_ERR(cudaGetLastError(),"devicemalloc");

    //========================================cpu st
    printf("===================================CPU\n");
    printf("Step 0: t = %.6f\n", time_cur);
    auto printtimestart = std::chrono::high_resolution_clock::now();

    while(time_cur<time_end){
        double dt = DT;
        if (time_cur+dt>time_end){
            dt = time_end-time_cur;
            time_cur = time_end;
        }

        cross_tstep(u1,u2,dt);
        std::swap(u1,u2);

        time_cur+=dt;
        step_cur++;

        if (step_cur%1000==0){
          double averdif = 0;
          for (int i=1; i<NX-1;i++){
          for (int j=1; j<NY-1;j++){
          averdif += abs(u1[i][j]-u2[i][j]);
          }
          }
          averdif /= (NX-2)*(NY-2);
          printf("it %d: averdif %.7f\n", step_cur , averdif);
        }

        if (step_cur>1e7){printf("steps oveflw\n"); return 0;}
    }

    auto printtimeended = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(printtimeended - printtimestart);
    printf("CPU Time: %.4f seconds\n", duration.count());
    printf("step_cur: %d\n", step_cur);

    //gpu strt========================================================
    printf("===================================GPU\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    init_border_1d(hostu1);
    init_border_1d(hostu2);

    cudaMemcpy(devsu1,hostu1,   NX*NY*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy( devsu2,hostu2,  NX*NY*sizeof(double),cudaMemcpyHostToDevice);
    CHECK_ERR(cudaGetLastError(),"h2d");

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE );
    dim3 dimGrid((NX +BLOCK_SIZE -1 )/BLOCK_SIZE,(NY +BLOCK_SIZE -1 )/BLOCK_SIZE );

    hosttime_cur = 0.0;
    hoststep_cur = 0;
    cudaEventRecord(start);
    printf("Step 0: t = %.6f\n", hosttime_cur);

    while (hosttime_cur < time_end) {
        double dt = DT;
        if (hosttime_cur + dt > time_end) {
            dt = time_end - hosttime_cur;
            hosttime_cur = time_end;
        } else {
            hosttime_cur += dt;
        }

        heat_step_kernel<<<dimGrid, dimBlock>>>(devsu1, devsu2, dt);
        CHECK_ERR(cudaGetLastError(), "kernel launch");

        double* temp = devsu1;
        devsu1 = devsu2;
        devsu2 = temp;
        hoststep_cur++;

        if (hoststep_cur > 100000) {
            printf("GPU steps overflow!\n");
            break;
        }
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(hostu1,devsu1,   NX*NY*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostu2,devsu2,   NX*NY*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK_ERR(cudaGetLastError(),"d2h");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gtime = -1;
    cudaEventElapsedTime(&gtime, start, stop);
    printf("GPU time: %.3f ms\n", gtime);
    printf("step_cur: %d\n", hoststep_cur);

    //======

    printf("\nCPU\n");
    print_matrix_2d(u2);
    printf("\nGPU\n");
    print_matrix_1d(hostu2);
    
    //======

    free_2_array(u1,NX);
    free_2_array(u2,NX);
    cudaFreeHost(hostu1);
    cudaFreeHost(hostu2);
    cudaFree(devsu1);
    cudaFree(devsu2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
};

