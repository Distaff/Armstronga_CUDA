#include <stdio.h>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <string>
#include "algorithms.h"

const intType_t MAX_RANGE = (intType_t)1099511627776 * (intType_t)1099511627776;	//2^40 * 2^40. Zapisanie 2^80 jest niemo¿liwe - literal moze byc maksymalnie 64-bitowy

const intType_t CHUNK_SIZE = 100000000;
const int SM_PER_GPU = 6;
const int THREADS_PER_SM = 786;

const int GPU_NUMBER = 1;	//Dla wielu kart graficznych

#ifdef INT128_
__device__ std::string uint128_to_string(intType_t val) {
	std::string res = "";
	char buff;
	while (val != 0) {
		buff = val % 10 + 48;	// Konwersja int na znak ascii
		res += buff;
		val = val / 10;
	}
	return res;
}
#endif


__global__ void calculate_chunk(intType_t range_min, intType_t range_max) {
	intType_t index = blockIdx.x * blockDim.x + threadIdx.x;
	intType_t stride = blockDim.x * gridDim.x;
	for (intType_t j = range_min + index; j < range_max; j += stride) {
		if (sum_p(j) == j) {
#ifdef INT64_
			//printf("FOUND: %-30" PRIu64 "%-30s\n", j, prime_check(j) ? "PRIME!" : "ODD");
			printf("FOUND: %-30" PRIu64 "%-30s\n", j, prime_check(j) ? "PRIME!" : "ODD");
#endif
#ifdef INT128_
			printf("FOUND: %-30s%-30s\n", uint128_to_string(j).c_str(), prime_check(j) ? "PRIME!" : "ODD");
#endif
		}
	}
}

__global__ void calculate_chunk_multi_GPU(intType_t range_min, intType_t range_max, int instance, int totalInstances) {
	intType_t index = blockIdx.x * blockDim.x + threadIdx.x;
	intType_t stride = blockDim.x * gridDim.x;
	index += instance * stride;
	stride = stride * totalInstances;
	for (intType_t j = range_min + index; j < range_max; j += stride) {
		if (sum_p(j) == j) {
#ifdef INT64_
			//printf("FOUND: %-30" PRIu64 "%-30s\n", j, prime_check(j) ? "PRIME!" : "ODD");
			printf("FOUND: %-30" PRIu64 "%-30s\n", j, prime_check(j) ? "PRIME!" : "ODD");
#endif
#ifdef INT128_
			printf("FOUND: %-30s%-30s\n", uint128_to_string(j).c_str(), prime_check(j) ? "PRIME!" : "ODD");
#endif
		}
	}
}

void run_multiple_GPU(intType_t range_min, intType_t range_max, int totalInstances) {
	for (int i = 0; i < totalInstances; i++) {
		cudaSetDevice(i);
		calculate_chunk_multi_GPU <<<SM_PER_GPU, THREADS_PER_SM>>> (range_min, range_max, i, totalInstances);
	}
}

void sync_multiple_GPU(int totalInstances) {
	for (int i = 0; i < totalInstances; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
}

__global__ void test_sqrt() {
	printf("TEST1: %-30s\n", uint128_to_string(sqrt(17)).c_str());
}

int main() {
	fill_lookup<<<1, 1>>>();

	auto err = cudaDeviceSynchronize();
	printf("GPUassert: %s\n", cudaGetErrorString(err));
	
	/*
	//Kod dla pojedynczego GPU:
	for (intType_t i = 3; i < UINT64_MAX; i += CHUNK_SIZE) {
		intType_t range_max = i + CHUNK_SIZE;
		calculate_chunk<<<SM_PER_GPU, THREADS_PER_SM>>> (i, range_max);

		auto err = cudaDeviceSynchronize();
		printf("GPUassert: %s\t", cudaGetErrorString(err));

		printf("FINISHED BLOCK: %" PRIu64 " - %" PRIu64 "\r", i, range_max - 1);
		range_max += CHUNK_SIZE;
	}*/
	
	
	//Kod dla wielu GPU:
	for (intType_t i = 3; i < MAX_RANGE; i += CHUNK_SIZE) {
		intType_t range_max = i + CHUNK_SIZE;
		run_multiple_GPU(i, range_max, GPU_NUMBER);

		sync_multiple_GPU(GPU_NUMBER);

		printf("FINISHED BLOCK: %s - %s\r", uint128_to_string(i).c_str(), range_max - 1);
		range_max += CHUNK_SIZE;
	}
}

