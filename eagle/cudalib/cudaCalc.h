#ifndef __EAGLE_CUDA_LIB
#define __EAGLE_CUDA_LIB

static const int _NT_CODES = 17;
static const int _READ_LENGTH = 100;
static const int ALIGNED_READ_LENGTH = (_READ_LENGTH+31)/32*32;
static const int SEQNT_MAP_NUM = 26;
typedef void* handle;

namespace Cpu {
	double calc_prob_region(const int* seq_map, const double *matrix, int read_length, const char *seq, int seq_length, int pos, int start, int end);
	double calc_prob_region_log_sum_exp(const int* seq_map, const double *matrix, int read_length, const char *seq, int seq_length, int pos, int start, int end);

};

namespace Gpu{
	handle create(int nProb, int nSeq, const int* seqnt_map, int seqnt_map_size, const double* matrix, int matrix_size);
	void   destroy(handle p);
	double calc_prob_region(handle p, int read_length, const char *seq, int seq_length, int start, int end);
	double calc_prob_region_log_sum_exp(handle p, int read_length, const char *seq, int seq_length, int start, int end);

};

#endif // __EAGLE_CUDA_LIB
