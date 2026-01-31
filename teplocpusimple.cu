# @title
%%file /tmp/t1v2.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <chrono>

#define NX 256
#define NY 256
#define DX (1.0/(NX-1))
#define DY (1.0/(NY-1))
#define DX2 (DX*DX)
#define DY2 (DY*DY)
#define C 1.0           // Коэффициент теплопроводности
#define DT (0.25 * fmin(DX*DX, DY*DY) / C) // Условие Куранта

#define NXX16 (NX*1/6)
#define NXX26 (NX*2/6)
#define NXX36 (NX*3/6)
#define NYY16 (NY*1/6)
#define NYY26 (NY*2/6)
#define NYY36 (NY*3/6)

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

void save_to_file(double** u, const char* filename, double t){
  FILE* fp = fopen(filename,"w");
  if(!fp){printf("fp fail\n"); return;}

  fprintf(fp, "# Time: %.6f\n", t);
  fprintf(fp, "# X Y U\n"        );

for(int i=0; i<NX; i++){
for(int j=0; j<NY; j++){
    fprintf(fp, "%.6f %.6f %.6f\n", i*DX, j*DY, u[i][j]);
}
}
  printf("Saved to %s\n", filename);
  fclose(fp);
}

int main(int argc, char* argv[]){
    double time_cur = 0.0;
    double time_end = 0.1;
    int step_cur = 0;

    printf("Solving heat equation until t = %.6f\n", time_end);
    printf("Grid: %dx%d, dx=%.6f, dy=%.6f\n", NX, NY, DX, DY);
    printf("Automatic time step: dt=%.8f (for stability)\n", DT);
    printf("Courant number: %.6f\n", C * DT * (1.0/(DX*DX) + 1.0/(DY*DY)));

    double** u1=allocate_2d_array(NX,NY);
    double** u2=allocate_2d_array(NX,NY);
    init_border(u1);
    init_border(u2);

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
    printf("Time: %.4f seconds\n", duration.count());
    //printf("Time: %.4f\n", printtimeended-printtimestart);
    printf("step_cur: %d\n", step_cur);

    printf("symmetry sliceoy\n");
    printf("%.6f %.6f %.6f\n",u1[NXX16][NYY16],u1[NXX16][NYY26],u1[NXX16][NYY36]);
    printf("%.6f %.6f %.6f\n",u1[NXX16][NY-1-NYY16],u1[NXX16][NY-1-NYY26],u1[NXX16][NY-NYY36]);
    printf("symmetry sliceox\n");
    printf("%.6f %.6f %.6f\n",u1[NXX16][NYY16],u1[NXX26][NYY16],u1[NXX36][NYY16]);
    printf("%.6f %.6f %.6f\n",u1[NX-1-NXX16][NYY16],u1[NX-1-NXX26][NYY16],u1[NX-NXX36][NYY16]);

    save_to_file(u1, "solution_final.txt", time_cur);

    free_2_array(u1,NX);
    free_2_array(u2,NX);
    return -0;
};

