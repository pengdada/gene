#include "cudaCalc.h"
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "myutility.h"

namespace Cpu {

	using namespace std;
	using namespace std::chrono;

	template<bool USE_BASELINE>
	inline double calc_read_prob(const int* seqnt_map, const double *matrix, int read_length, const char *seq, int seq_length, int pos, int baseline) {
		const int NT_CODES = 17;

#define clean_errno() (errno == 0 ? "None" : "fatal_error"/*strerror(errno)*/)
#define exit_err(M, ...) fprintf(stderr, "ERROR: (%s:%d: errno: %s) " M "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__); exit(EXIT_FAILURE)

		int i, c; // array[width * row + col] = value
		double probability = 0;
		for (i = pos; i < pos + read_length; i++) {

			if (i < 0) continue;
			if (i >= seq_length) 
				break;

			c = seq[i] - 'A';
			if (c < 0 || c >= 26) { exit_err("Character %c at pos %d (%d) not in valid alphabet\n", seq[i], i, seq_length); }

			probability += matrix[NT_CODES * (i - pos) + seqnt_map[c]];
			if (USE_BASELINE == 1) {
				if (probability < baseline - 10) break; // stop if less than 1% contribution to baseline (best/highest) probability mass
			}
		}
		return probability;
	}

	double calc_prob_region(const int* seq_map, const double *matrix, int read_length, const char *seq, int seq_length, int pos, int start, int end) {
		double probability = 0;
		if (start < 0) start = 0;
#if 0
		if (dp) {
			end += read_length;
			if (end >= seq_length) end = seq_length - 1;
			probability = smith_waterman_gotoh(matrix, read_length, seq, seq_length, start, end);
		}
		else
#endif
		{	
			const int USE_BASELINE = 1;
			probability = calc_read_prob<USE_BASELINE>(seq_map, matrix, read_length, seq, seq_length, pos, -1e6);
			double baseline = probability;
#ifdef _DEBUG
			std::vector<double> vec;
			load_data("./pMem.vec.prob.bin", vec);
#endif
			int i;
			for (i = start; i < end; i++) {
				if (i >= seq_length) break;
				if (i != pos) {

					double val = calc_read_prob<USE_BASELINE>(seq_map, matrix, read_length, seq, seq_length, i*read_length, baseline);
					//assert(IsEqual(vec[i], val) == 0);
					probability = log_add_exp(probability, val);
					if (probability > baseline) baseline = probability;
				}
			}
		}
		return probability;
	}

	double calc_prob_region_log_sum_exp(const int* seq_map, const double *matrix, int read_length, const char *seq, int seq_length, int pos, int start, int end) {
		double probability = 0;
		assert(start == 0);
#if 0
		if (dp) {
			end += read_length;
			if (end >= seq_length) end = seq_length - 1;
			probability = smith_waterman_gotoh(matrix, read_length, seq, seq_length, start, end);
		}
		else
#endif
		{
			std::vector<double> tmp(end);
			double* pBuf = &tmp[0];
			int i;
			const int USE_BASELINE = 0;
			for (i = start; i < end; i++) {
				pBuf[i] = calc_read_prob<USE_BASELINE>(seq_map, matrix, read_length, seq, seq_length, i*read_length, std::numeric_limits<double>::min());
			}
			probability = log_sum_exp(pBuf, end);
		}
		return probability;
	}
}