%%file 1.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

//#define MYPI 3,1415926535
#define MYPI 3.141f
//#define NX 256
#define NX 10
//#define NY 256
#define NY 10
#define DX (1.0/(NX-1))
#define DY (1.0/(NY-1))
#define DX2 ((double)(DX*DX))
#define DY2 ((double)(DY*DY))
//#define COURANT 1
//#define DT (0.25 * fmin(DX2,DY2)/COURANT )
#define MAXLOOPC 10000001
#define EPS 1e-8


#define BLOCKSIZE 16

//#define DODEBUG 1

#ifdef DODEBUG
#define CHECKERR(err, a){\
if (cudaSuccess!=err){\
printf("err  %s %d %s",__FILE__, __Line__, a);\
}}
#else
#define  CHECKERR(err){}
#endif



double* allocate_arr(double* a, int width, int heigh);
void delete_arr(double* a, int width, int heigh);
void cpu_borderinit(double* a, int width, int heigh);
void cpu_printarr(double* a, int width, int heigh);
double cpu_l3(double* a, double* b, int width, int heigh);
void cpu_jakobi_step(double* u, double* u_new, int width, int heigh);

/*
__global__ void gpu_borderinit(double* a, double* b, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int idx = y * width + x;

	if (x == 0) {
		a[idx] = b[idx] = 0.0;
	}
	else if (x == width - 1) {
		//a[idx] = b[idx] = sin(MYPI * (double)(width * y));
		a[idx] = b[idx] = sinf(3.1415926535 * (double)(width * y));
	}
	else if (y == 0) {
		a[idx] = b[idx] = 0.0;
	}
	else if (y == height - 1) {
		//a[idx] = b[idx] = sin(MYPI * (double)(height * x));
		a[idx] = b[idx] = sinf(3.1415926535 * (double)(height * x));
	}
}
*/

__global__ void gpu_jakobi_step(double* u, double* u_new, int width, int heigh) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= heigh) return;

	int indx = y * width + x;
	double physical_x = x * (1.0/(width-1)); // = x * DX
	double physical_y = y * (1.0/(heigh-1)); // = y * DY

	//double factor = 0.5 * (1.0 / ((1.0 / DX2) + (1.0 / DY2)));


	//BORDER
	//if (x == 0) { u_new[indx] = 0; }
	//else if (x == (int)(width - 1)) { u_new[indx] = sin((MYPI * (double)((double)width * (double)y))); }
	//else if (y == 0) { u_new[indx] = 0; }
	//else if (y == (int)(heigh - 1)) { u_new[indx] = sin((MYPI * (double)((double)heigh * (double)x))); }
	
	if (x == 0) { u_new[indx] = 0; }
	else if (x == (int)(width - 1)) {u_new[indx] =  sinf(MYPI * physical_y); }
	else if (y == 0) { u_new[indx] = 0; }
	else if (y == (int)(heigh - 1)) {u_new[indx] =  sinf(MYPI * physical_x); }
	else {        // Interior points
        int left  = y * width + (x - 1);
        int right = y * width + (x + 1);
        int down  = (y - 1) * width + x;
        int up    = (y + 1) * width + x;
        
        double denom = 2.0/DX2 + 2.0/DY2;
        u_new[indx] = ((u[left] + u[right]) / DX2 + (u[down] + u[up]) / DY2) / denom;
	}
	/*
	//INNER
	if (x >= width - 1 || y >= heigh - 1) return;

	int indxR = y * width + (x + 1);
	int indxL = y * width + (x - 1);
	int indxT = (y + 1) * width + x;
	int indxB = (y - 1) * width + x;

	//double laplc = (u[i - 1][j] + u[i + 1][j]) / DX2 + (u[i][j - 1] + u[i][j + 1]) / DY2;
	double laplc = (u[indxL] + u[indxR]) / DX2 + (u[indxB] + u[indxT]) / DY2;
	//printf(" lllll %.8f  ", factor);

	//

	//u_new[indx] = u[indx] + factor * laplc;
	u_new[indx] = factor * laplc;
	//printf("   %.8f  ", u_new[indx]);
	*/

}


int main() {

	//double* arr1; arr1 = allocate_arr(&arr1, NX, NY);
	//double* arr2; arr2 = allocate_arr(&arr2, NX, NY);
	double* arr1 = (double*)calloc(NX * NY, sizeof(double));
	double* arr2 = (double*)calloc(NX * NY, sizeof(double));
	double* h_arr1;
	double* h_arr2;
	double* d_arr1;
	double* d_arr2;
	cudaMallocHost(&h_arr1, (NX * NY * sizeof(double)));
	cudaMallocHost(&h_arr2, (NX * NY * sizeof(double)));
	cudaMalloc(&d_arr1, (NX * NY * sizeof(double)));
	cudaMalloc(&d_arr2, (NX * NY * sizeof(double) )   );
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//CPU

	auto starttcpu = std::chrono::steady_clock::now();

	cpu_borderinit(arr1, NX, NY);
	cpu_borderinit(arr2, NX, NY);
	//cpu_printarr(arr1, NX, NY);
	//cpu_printarr(arr2, NX, NY);

	/*
	if (NX < 30) {
		printf("arr1\n\n");
		cpu_printarr(arr1, NX, NY);
		printf("arr2\n\n");
		cpu_printarr(arr2, NX, NY);
	}
	*/

	{
	//double printme = cpu_l3(arr1, arr2, NX, NY);
	//printf("\n\n%.8f\n\n", printme);
	//cpu_jakobi_step(arr1, arr2, NX, NY);
	//printf("\n\n%.8f\n\n", printme);
	}

	for (int step = 0; step < MAXLOOPC; step++) {

		cpu_jakobi_step(arr1, arr2, NX, NY);
		std::swap(arr1, arr2);
		double residual = cpu_l3(arr1, arr2, NX, NY);

		if ((step % 1000) == 0) {
			printf(" it %d:  %.8f\n", step , residual);
		}

		if (residual < EPS) {
			printf("Achived residual < %.10f on step %d ",  EPS, step);
			break;
		}


	}

	if (NX < 30) {
		//printf("\narr1\n\n");
		//cpu_printarr(arr1, NX, NY);
		printf("\narr2\n\n");
		cpu_printarr(arr2, NX, NY);
	}

	auto endtcpu = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedcpu = endtcpu - starttcpu;
	std::cout <<  "\n"	<< "Time: " << elapsedcpu.count();

	//GPU
	printf("\n\n=================gpu\n\n");

 	cpu_borderinit(h_arr1, NX, NY);
	cpu_borderinit(h_arr2, NX, NY);

	//cudaMemcpy(d_arr1, &h_arr1, NX*NY*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr1, h_arr1, NX*NY*sizeof(double), cudaMemcpyHostToDevice);
	CHECKERR("H2D");
	//cudaMemcpy(d_arr2, &h_arr2, NX * NY * sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, h_arr2, NX * NY * sizeof(double), cudaMemcpyHostToDevice);
	CHECKERR("H2D");

 
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid((NX + BLOCKSIZE - 1) / BLOCKSIZE,
							(NY + BLOCKSIZE - 1) / BLOCKSIZE);
	//dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
//dim3 dimGrid((NX + BLOCKSIZE - 1) / BLOCKSIZE,
  //           (NY + BLOCKSIZE - 1) / BLOCKSIZE);

	//gpu_borderinit<<<dimGrid, dimBlock >>>(d_arr1, d_arr2,NX, NY);
	//CHECKERR("gpu_borderinit");

	double gpu_residual = 1.0;
	for (int step = 0; step < MAXLOOPC; step++) {
			gpu_jakobi_step<<<dimGrid, dimBlock>>>(d_arr1, d_arr2, NX, NY);
				
			std::swap(d_arr1, d_arr2);
			std::swap(h_arr1, h_arr2);


			if (step % 10 == 0) {
					cudaMemcpy(h_arr1, d_arr1, NX*NY*sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(h_arr2, d_arr2, NX*NY*sizeof(double), cudaMemcpyDeviceToHost);
					gpu_residual = cpu_l3(h_arr1, h_arr2, NX, NY); // Reuse CPU residual function
					
					if (gpu_residual < EPS) {
							printf("GPU converged at step %d\n", step);
							break;
        }
    }
}





 //gpu end
	if (NX < 30) {
		cudaMemcpy(h_arr1, &d_arr1, NX * NY * sizeof(double*), cudaMemcpyDeviceToHost);
		CHECKERR("D2H");
		cudaMemcpy(h_arr2, &d_arr2, NX * NY * sizeof(double*), cudaMemcpyDeviceToHost);
		CHECKERR("D2H");

		//printf("arr1\n\n");
		//cpu_printarr(h_arr1, NX, NY);
		printf("arr2\n\n");
		cpu_printarr(h_arr2, NX, NY);
	}

	//del
	//cudaFree(h_arr1);
	//cudaFree(h_arr2);
	cudaFreeHost(h_arr1);
	cudaFreeHost(h_arr2);
	cudaFree(d_arr1);
	cudaFree(d_arr2);
	//cudaFree(start);
	//cudaFree(stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	delete_arr(arr1, NX, NY);
	delete_arr(arr2, NX, NY);

};

/*
double* allocate_arr(double* a, int width, int heigh) {
	a = (double*)malloc( (width * heigh * sizeof(double*)) );
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = i * width + j;
			//a[indx] = 0;
			//a[indx] = (double)calloc(1, sizeof(double));
			a[indx] = calloc(1, sizeof(double));
			//a[i] = (double*)calloc(NY, sizeof(double));
			//a[i] = calloc(NY, sizeof(double));
		}
	}
	return a;
}
*/

double* allocate_arr(double* a, int width, int heigh) {
	a = (double*)calloc( (width * heigh) , sizeof(double*));
	return a;
}

void delete_arr(double* a, int width, int heigh) {
	//for (int i = 0; i < NX; i++) {
	//	for (int j = 0; j < NY; j++) {
	//		free(a[i]);
	//	}
	//}
	free(a);
}

void cpu_borderinit(double* a, int width, int heigh) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = j * width + i;

			if (i == 0) { a[indx] = 0; }
			else if (i == (int)(width - 1)) { a[indx] = sin((MYPI* (double)((double)DX * (double)j)   ) ) ; }
			else if (j == 0) { a[indx] = 0; }
			else if ( j == (int)(heigh-1)) { a[indx] = sin( (MYPI * (double) (    (double)DY * (double)i   ) ) ) ; }
		}
	}
}


void cpu_printarr(double* a, int width, int heigh) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = j * width + i;
			printf("%.5f ", a[indx]);
		}
		printf("\n");
	}

	/*
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = j * width + i;

			if (i == 0) { printf("%.1f ", a[indx]); }
			else if (i == (width - 1)) { printf("%.1f ", a[indx]); }
			else if (j == 0) { printf("%.1f ", a[indx]); }
			else if (j == (heigh - 1)) { printf("%.1f ", a[indx]); }
		}
	}
	*/
}

double cpu_l3(double* a, double* b, int width, int heigh) {
	double sum = 0.0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = j * width + i;
			sum += pow(  (fabs(a[indx] - b[indx])) , 3.0   );
		}
	}
	//printf("%.10f \n", sum);
	//sum = pow(sum, (1.0 / 3.0));
	sum = pow(sum , (0.333333) );
	return sum;
}


void cpu_jakobi_step(double* u, double* u_new, int width, int heigh) {
	double factor = 0.5 * (1.0 / ((1.0 / DX2) + (1.0 / DY2)));
	//printf("ffff %.8f  ", factor);
	//printf("ffff %.8f  ", (1.0 / DX2));
	//printf("ffff %.8f  ", (1.0 / DY2));
	//printf("ffff %.8f  ", ((1.0 / DX2) + (1.0 / DY2)));


	//BORDER
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < heigh; j++) {
			int indx = j * width + i;

			if (i == 0) { u_new[indx] = 0; }
			else if (i == (int)(width - 1)) { u_new[indx] = sin((MYPI * (double)((double)DX * (double)j))); }
			else if (j == 0) { u_new[indx] = 0; }
			else if (j == (int)(heigh - 1)) { u_new[indx] = sin((MYPI * (double)((double)DY * (double)i))); }
		}
	}
	//INNER
	for (int i = 1; i < width-1; i++) {
		for (int j = 1; j < heigh-1; j++) {
			int indx = j * width + i;
			int indxR = j * width + (i+1);
			int indxL = j * width + (i-1);
			int indxT = (j+1) * width + i;
			int indxB = (j-1) * width + i;

			//double laplc = (u[i - 1][j] + u[i + 1][j]) / DX2 + (u[i][j - 1] + u[i][j + 1]) / DY2;
			double laplc = (u[indxL] + u[indxR]) / DX2 + (u[indxB] + u[indxT]) / DY2;
			//printf(" lllll %.8f  ", factor);

			//

			//u_new[indx] = u[indx] + factor * laplc;
			u_new[indx] = factor * laplc;
			//u_new = ((uL + uR)/DX2 + (uB + uT)/DY2) / (2.0/DX2 + 2.0/DY2);
			//printf("   %.8f  ", u_new[indx]);
		}
	}

}





