#ifndef __MYUTILITY_H
#define __MYUTILITY_H
#include <math.h>
#include <vector>
#include <assert.h>

#define MAX2(x, y) (x) >= (y)?(x):(y)
#define MIN2(x, y) (x) <= (y)?(x):(y)

double log_add_exp(double a, double b);

inline double log_add_exp(double a, double b) {
	double max_exp = a > b ? a : b;
	return log(exp(a - max_exp) + exp(b - max_exp)) + max_exp;
}

template<typename T> inline
void Transpose(const T* src, int src_width, int src_height, T* dst, int dst_width, int dst_height) {
	int _dst_height = MIN2(dst_height, src_width);
	int _dst_width = MIN2(dst_width, src_height);
#pragma omp parallel for	
	for (int y = 0; y < _dst_height; y++) {
		T* pDst = dst + dst_width * y;
		const T* pSrc = src + y;
		for (int x = 0; x < _dst_width; x++) {
			*pDst = *pSrc;
			pDst++;
			pSrc += src_width;
		}
	}
}

//template<> static
//static void Transpose<float>(const float* src, int src_width, int src_height, float* dst, int dst_width, int dst_height)
//{
//#if 0
//	__m128 mm0, mm1, mm2, mm3;
//
//	const int src_width_4 = src_width << 2;
//	for (int i = 0; i < dst_height; i += 4) {
//		float* p0 = dst + dst_width * i;
//		float* p1 = p0 + dst_width;
//		float* p2 = p1 + dst_width;
//		float* p3 = p2 + dst_width;
//
//		const float* ps0 = src + i;
//		const float* ps1 = ps0 + src_width;
//		const float* ps2 = ps1 + src_width;
//		const float* ps3 = ps2 + src_width;
//		for (int j = 0; j<dst_width; j += 4, p0 += 4, p1 += 4, p2 += 4, p3 += 4, ps0 += src_width_4, ps1 += src_width_4, ps2 += src_width_4, ps3 += src_width_4) {
//			mm0 = _mm_load_ps(ps0);
//			mm1 = _mm_load_ps(ps1);
//			mm2 = _mm_load_ps(ps2);
//			mm3 = _mm_load_ps(ps3);
//			_MM_TRANSPOSE4_PS(mm0, mm1, mm2, mm3);
//			_mm_store_ps(p0, mm0);
//			_mm_store_ps(p1, mm1);
//			_mm_store_ps(p2, mm2);
//			_mm_store_ps(p3, mm3);
//		}
//	}
//#else
//	for (int y = 0; y < dst_height; y++) {
//		float* pDst = dst + dst_width * y;
//		const float* pSrc = src + y;
//		for (int x = 0; x < dst_width; x++) {
//			*pDst = *pSrc;
//			pDst++;
//			pSrc += src_width;
//		}
//	}
//#endif
//}

static long get_file_size(FILE* fp) {
	if (fp == NULL) {
		return 0;
	}
	int pos = ftell(fp);
	if (fseek(fp, 0, SEEK_END) != 0) {
		return 0;
	}
	int file_size = ftell(fp);
	if (file_size == -1) {
		return 0;
	}
	if (fseek(fp, pos, SEEK_SET) != 0) {
		return 0;
	}
	return file_size;
}

template<typename T> inline
int load_data(const char* path, std::vector<T>& vec) {
	FILE* fp = fopen(path, "rb");
	int nrd = 0;
	int size = 0;
	if (fp) {
		size = get_file_size(fp);
		vec.resize(size / sizeof(T));
		if (vec.size() > 0) {
			nrd = fread(&vec[0], 1, size, fp);
		}
		fclose(fp);
	}
	int nRet = (nrd > 0 && nrd == size) ? 0 : 1;
	printf("load_data:%s, %d,, %s\n", path, nrd, nRet==0?"success":"failed");
	return nRet;
}
static int write_file(const char* path, const void* data, unsigned int size) {
	FILE* fp = fopen(path, "wb");
	unsigned int len = 0;
	if (fp) {
		len = fwrite(data, 1, size, fp);
		fclose(fp);
	}
	int nRet = (len > 0 && len == size) ? 0 : 1;
	printf("write_file:%s, %d, %s\n", path, len, nRet == 0 ? "success" : "failed");
	return nRet;
}

#define ABS(x) ((x) >= 0?(x):(-(x)))


inline int IsEqual(int x, int y) {
	return x == y ? 0 : 1;
}
inline int IsEqual(double x, double y) {
	return ABS(x - y) < 0.00000001 ? 0 : 1;
}

#endif // !__MYUTILITY_H
