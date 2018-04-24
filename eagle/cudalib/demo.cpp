#include <stdio.h>
#include <iostream>
#include "cudaCalc.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <string.h>
#include "myutility.h"
using namespace std;
using namespace std::chrono;


void init_seqnt_map(int *seqnt_map) {
	/* Mapping table, symmetrical according to complement */
	memset(seqnt_map, 0, sizeof(int) * 26);

	seqnt_map['A' - 'A'] = 0;
	seqnt_map['C' - 'A'] = 1;

	/* Ambiguous codes */
	seqnt_map['H' - 'A'] = 2; // A, C, T
	seqnt_map['B' - 'A'] = 3; // C, G, T
	seqnt_map['R' - 'A'] = 4; // A, G
	seqnt_map['K' - 'A'] = 5; // G, T
	seqnt_map['S' - 'A'] = 6; // G, C
	seqnt_map['W' - 'A'] = 7; // A, T

	seqnt_map['N' - 'A'] = 8;
	seqnt_map['X' - 'A'] = 8;

	// W also in 9, S also in 10
	seqnt_map['M' - 'A'] = 11; // A, C
	seqnt_map['Y' - 'A'] = 12; // C, T
	seqnt_map['V' - 'A'] = 13; // A, C, G
	seqnt_map['D' - 'A'] = 14; // A, G, T

	seqnt_map['G' - 'A'] = 15;
	seqnt_map['T' - 'A'] = 16;
	seqnt_map['U' - 'A'] = 16;
}

void TestCpu()
{
	std::cout << "TestCpu: use single CPU for computation" << std::endl;

	std::vector<double> vec_matrix;
	std::vector<char> vec_seq;
	std::vector<int>  vec_seq_map;

	vec_seq_map.resize(26);
	init_seqnt_map(&vec_seq_map[0]);
	
	//const char path_matrix[] = "../eagle/cudalib/args/matrix.bin";
	//const char path_seq[] = "../eagle/cudalib/args/seq.bin";

	const char path_matrix[] = "./args/matrix.bin";
	const char path_seq[] = "./args/seq.bin";

	load_data(path_matrix, vec_matrix);
	load_data(path_seq, vec_seq);

	const int* seq_map = &vec_seq_map[0];
	double *matrix = 0;
	int read_length = _READ_LENGTH;
	const char *seq = 0;
	int seq_length = vec_seq.size();
	int pos=0;
	int start = 0;
	int end = seq_length / read_length;
	if (vec_matrix.size() > 0) {
		matrix = &vec_matrix[0];
	}
	if (seq_length > 0) {
		seq = &vec_seq[0];
	}
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//double prob= Cpu::calc_prob_region(seq_map, matrix, read_length, seq, seq_length, pos, start, end);
	double prob = Cpu::calc_prob_region_log_sum_exp(seq_map, matrix, read_length, seq, seq_length, pos, start, end);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	long long duration = duration_cast<milliseconds>(t2 - t1).count();
	
	printf("matrix_size = %dx%d\n", _READ_LENGTH, _NT_CODES);
	printf("seq_length = %d\n", seq_length);
	std::cout << "probablility = " << prob << std::endl;
	std::cout << "duration time : " << duration <<" ms"<<std::endl;
	std::cout << "------------------------------------------" << std::endl;
}

void TestGpu()
{
	std::cout << "TestGpu: use single GPU for computation" << std::endl;

	std::vector<double> vec_matrix;
	std::vector<char> vec_seq;
	std::vector<int>  vec_seq_map;

	vec_seq_map.resize(26);
	init_seqnt_map(&vec_seq_map[0]);

	//const char path_matrix[] = "../eagle/cudalib/args/matrix.bin";
	//const char path_seq[] = "../eagle/cudalib/args/seq.bin";

	const char path_matrix[] = "./args/matrix.bin";
	const char path_seq[] = "./args/seq.bin";

	load_data(path_matrix, vec_matrix);
	load_data(path_seq, vec_seq);

	const int* seq_map = &vec_seq_map[0];
	double *matrix = 0;
	int read_length = _READ_LENGTH;
	const char *seq = 0;
	int seq_length = vec_seq.size();
	int pos = 0;
	int start = 0;
	int end = seq_length / read_length;
	if (vec_matrix.size() > 0) {
		matrix = &vec_matrix[0];
	}
	if (seq_length > 0) {
		seq = &vec_seq[0];
	}

	handle hdl = Gpu::create(seq_length / _READ_LENGTH, seq_length, seq_map,  vec_seq_map.size(), matrix, vec_matrix.size());

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//double prob = Gpu::calc_prob_region(hdl, read_length, seq, seq_length, start, end);
	double prob = Gpu::calc_prob_region_log_sum_exp(hdl, read_length, seq, seq_length, start, end);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	long long duration = duration_cast<milliseconds>(t2 - t1).count();

	Gpu::destroy(hdl);

	printf("matrix_size = %dx%d\n", _READ_LENGTH, _NT_CODES);
	printf("seq_length = %d\n", seq_length);
	std::cout << "probablility = " << prob << std::endl;
	std::cout << "duration time : " << duration << " ms" << std::endl;
	std::cout << "------------------------------------------" << std::endl;
}


int main(int argc, char** argv) {
	TestCpu();
	TestGpu();
	return 0;
}
