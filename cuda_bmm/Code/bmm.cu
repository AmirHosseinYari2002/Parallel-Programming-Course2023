//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y

#define bx blockIdx.x
#define by blockIdx.y

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 32
#define TILEY 16

// you may define other parameters here!
// it's lower than tilex and tiley or bigger than both of them
#define TILE 128
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n / TILEX, n / TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX, TILEY);
	return dimBlock;
}

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

	__shared__ float shared_ad[TILEY][TILE + 1]; 
	__shared__ float shared_bd[TILE + 1][TILEX];

	int global_row = by * TILEY + ty;
	int global_column = bx * TILEX + tx;
	float result = 0;

	for (int i = 0; i < n / TILE; ++i) {
		// Load tiles into shared memory with coalesced access for mad
		if (tx < TILE && ty < TILEY) {
			shared_ad[ty][tx] = ad[global_row * n + i * TILE + tx];
		}

		// Load tiles into shared memory with coalesced access for mbd
		if (tx < TILEX && ty < TILE) {
			shared_bd[ty][tx] = bd[(i * TILE + ty) * n + global_column];
		}

		__syncthreads();

		// Compute partial sum with coalesced access for both mad and mbd
		for (int k = 0; k < TILE; ++k) {
			result += shared_ad[ty][k] * shared_bd[k][tx];
		}

		__syncthreads();
	}
	cd[global_row * n + globalCol] = result;
}
