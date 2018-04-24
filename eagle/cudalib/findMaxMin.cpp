#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
//#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <chrono>
#include <string.h>
#ifdef _WIN32
#pragma comment(lib, "cublas.lib")
#endif

typedef float Real;

void findMaxAndMinGPU(Real* values, int* max_idx, int* min_idx, int n)
{
	Real* d_values;
	cublasHandle_t handle;
	cublasStatus_t stat;
	cudaMalloc((void**)&d_values, sizeof(Real) * n);
	cudaMemcpy(d_values, values, sizeof(Real) * n, cudaMemcpyHostToDevice);
	cublasCreate(&handle);

	stat = cublasIsamax(handle, n, d_values, 1, max_idx);
	if (stat != CUBLAS_STATUS_SUCCESS)
		printf("Max failed\n");

	stat = cublasIsamin(handle, n, d_values, 1, min_idx);
	if (stat != CUBLAS_STATUS_SUCCESS)
		printf("min failed\n");

	cudaFree(d_values);
	cublasDestroy(handle);
}

int main0(void)
{
	const int vmax = 1000, nvals = 10000;

	float vals[nvals];
	srand(time(NULL));
	for (int j = 0; j<nvals; j++) {
		//vals[j] = float(rand() % vmax);
		vals[j] = nvals - j - 1;
	}

	int minIdx, maxIdx;
	findMaxAndMinGPU(vals, &maxIdx, &minIdx, nvals);

	int cmin = 0, cmax = 0;
	for (int i = 1; i<nvals; i++) {
		cmin = (vals[i] < vals[cmin]) ? i : cmin;
		cmax = (vals[i] > vals[cmax]) ? i : cmax;
	}

	fprintf(stdout, "%d %d %d %d\n", minIdx, cmin, maxIdx, cmax);

	return 0;
}