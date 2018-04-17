#include "cudaCalc.h"
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include "cudaLib.h"
#include "myutility.h"
#include <chrono>
#include <string.h>
using namespace std;
using namespace std::chrono;

typedef unsigned int uint;
static const int WarpSize = 32;
namespace Gpu {
	__constant__ double c_matrix[_NT_CODES][ALIGNED_READ_LENGTH];
	__constant__ int c_seqnt_map[SEQNT_MAP_NUM];
	
	__device__ inline
		double shfl_down(double var, unsigned int srcLane, int width = 32) {
		int2 a = *reinterpret_cast<int2*>(&var);
#if CUDART_VERSION >= 9000
		a.x = __shfl_down_sync(0xFFFFFFFF, a.x, srcLane, width);
		a.y = __shfl_down_sync(0xFFFFFFFF, a.y, srcLane, width);
#else
		a.x = __shfl_down(a.x, srcLane, width);
		a.y = __shfl_down(a.y, srcLane, width);
#endif
		return *reinterpret_cast<double*>(&a);
	}

	__global__ void kernel_calc_prob_region(double* probability, const char* __restrict seq, uint seq_length, uint read_length, uint count)
	{
		const uint landId = threadIdx.x & 31;
		const uint warpId = threadIdx.x / WarpSize;
		const uint blockWarpCount = blockDim.x / WarpSize;
		const uint widx = blockIdx.x*blockWarpCount + warpId;
		const uint total_warp = blockWarpCount * gridDim.x;
		
		//__shared__ double smem[_NT_CODES][ALIGNED_READ_LENGTH];
		//if (widx == 0 && threadIdx.x == 0) {
		//	for (int i = 0; i < _NT_CODES; i++) {
		//		for (int j = 0; j < ALIGNED_READ_LENGTH; j++) {
		//			smem[i][j] = c_matrix[i][j];
		//		}
		//	}
		//}
		//__syncthreads();
#if 1
		for (uint i = widx; i < count; i += total_warp) {
			double sum = 0;
			const char* cur_seq = seq + i * read_length;
			for (uint j = landId; j < read_length; j += WarpSize) {
				int c = cur_seq[j] - 'A';			
				assert(c >= 0 && c < 26);
				int d = c_seqnt_map[c];
				sum += c_matrix[d][j];
			}
#pragma unroll
			for (int offset = WarpSize / 2; offset > 0; offset /= 2)
				sum += shfl_down(sum, offset);
			if (landId == 0) {
				probability[i] = sum;
			}
		}
#endif
	}
	struct Mem{
		Mem(int nProb, int nSeq) {
			devProb.MallocBuffer(nProb, 1, 1);
			devSeq.MallocBuffer(nSeq, 1, 1);
			vec_matrix.resize(_NT_CODES*ALIGNED_READ_LENGTH);
			vec_prob.resize(nProb);
		}
		DevData<double> devProb;
		DevData<char>   devSeq;
		std::vector<double> vec_matrix;
		std::vector<double> vec_prob;
	};
	handle create(int nProb, int nSeq,  const int* seqnt_map, int seqnt_map_size, const double* matrix, int matrix_size) {
		Mem* p = new Mem(nProb, nSeq);
		{
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			{
				Transpose(matrix, _NT_CODES, _READ_LENGTH, &p->vec_matrix[0], ALIGNED_READ_LENGTH, _NT_CODES);
			}
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			long long duration = duration_cast<microseconds>(t2 - t1).count();
			std::cout << "Transpose duration = " << duration*0.001 <<" ms"<<std::endl;
		}
		{
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			{
				LoadConstant(&c_matrix[0][0], &p->vec_matrix[0], p->vec_matrix.size());
				LoadConstant(c_seqnt_map, seqnt_map, sizeof(c_seqnt_map)/sizeof(seqnt_map[0]));
			}
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			long long duration = duration_cast<microseconds>(t2 - t1).count();
			std::cout << "LoadConstant c_matrix and c_seqnt_map, duration = " << duration * 0.001 << " ms" << std::endl;
		}
		return (handle)p;
	}
	void destroy(void* p) {
		if (p) {
			delete ((Mem*)p);
		}
	}
	double calc_prob_region(handle hdl, int read_length, const char *seq, int seq_length, int start, int end) {
		Mem* pMem = (Mem*)hdl;
		double probability = 0;
		assert(start == 0);
		assert(end > start);
		dim3 threads(1024, 1, 1);
		int warps = threads.x / WarpSize;
		dim3 blocks(UpDivide(end, warps), 1, 1);
		pMem->devSeq.CopyFromHost(seq, seq_length, seq_length, 1, 1);
		kernel_calc_prob_region<<<blocks, threads >>>(pMem->devProb.GetData(), pMem->devSeq.GetData(), seq_length, read_length, end-start);
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;
		pMem->devProb.CopyToHost(&pMem->vec_prob[0], pMem->vec_prob.size(), pMem->vec_prob.size(), 1, 1);
#ifdef _DEBUG
		write_file("./pMem.vec.prob.bin", &pMem->vec_prob[0], pMem->vec_prob.size()*sizeof(pMem->vec_prob[0]));
#endif // _DEBUG
		{
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			{
				double* pProb = &pMem->vec_prob[0];
				probability = pProb[0];
				for (int i = start + 1; i < end; i++) {
					{
						probability = log_add_exp(probability, pProb[i]);
						//if (probability > baseline) baseline = probability;
					}
				}
			}
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			long long duration = duration_cast<milliseconds>(t2 - t1).count();
			std::cout << "log_add_exp duration = " << duration << std::endl;
		}
		return probability;
	}
}