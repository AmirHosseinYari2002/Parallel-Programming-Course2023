//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "scan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z


#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define gdx gridDim.x
#define gdy gridDim.y
#define gdz gridDim.z


// __global__ void kernelFunc( uint8_t* cb , const uint8_t* const ab , const uint8_t* const A , int i , const int n , uint8_t* alpha_to_g , uint8_t* index_of_g )
// {
//     int num = (by*gdx*bdx)+(bx*bdx)+(tx);
//     __shared__ uint8_t alpha_to_h[256];
//     __shared__ uint8_t index_of_h[256];
//     __shared__ uint8_t AB[16];
//     if( tx< 256)
//     {
//         alpha_to_h[tx] = alpha_to_g[tx];
//         index_of_h[tx] = index_of_g[tx];
//         if((tx) <16){
//             AB[tx] = A[tx];
//         }
//     }
//     __syncthreads();

//     if( num <  (4 * n) && ( num >= 4*(1<<(i)) )) 
//     {
//         int k = (num % 4);
//         int vector = (num - 4*(1<<(i))) - k;
//         uint8_t c[4];
//         for(int s = 0 ; s < 4 ; ++s)
//         {
//             c[s] = ab[(vector + s)];
//         }
//         uint8_t r = 0;
//         if( AB[k*4 + 0]!= 0 && c[0]!= 0){
//             r ^= alpha_to_h[(uint32_t(index_of_h[c[0]]) + uint32_t(index_of_h[AB[k * 4 + 0]]))%255];
//         }
//         if( AB[k*4 + 1]!= 0 && c[1]!= 0){
//             r ^= alpha_to_h[(uint32_t(index_of_h[c[1]]) + uint32_t(index_of_h[AB[k * 4 + 1]]))%255];
//         }
//         if( AB[k*4 + 2]!= 0 && c[2]!= 0){
//             r ^= alpha_to_h[(uint32_t(index_of_h[c[2]]) + uint32_t(index_of_h[AB[k * 4 + 2]]))%255];
//         }
//         if( AB[k*4 + 3]!= 0 && c[3]!= 0){
//             r ^= alpha_to_h[(uint32_t(index_of_h[c[3]]) + uint32_t(index_of_h[AB[k * 4 + 3]]))%255];
//         }
//         cb[num]  ^= r;
//     }
// }

__global__ void kernelFunc( uint8_t* cipherBlock , const uint8_t* const alphaBlock , const uint8_t* const inputArray , int iteration , const int arraySize , uint8_t* alphaToGlobal , uint8_t* indexOfGlobal )
{
    int globalIndex = (blockIdx.y*gridDim.x*blockDim.x)+(blockIdx.x*blockDim.x)+(threadIdx.x);
    __shared__ uint8_t alphaToShared[256];
    __shared__ uint8_t indexOfShared[256];
    __shared__ uint8_t inputBlock[16];
    if( threadIdx.x < 256)
    {
        alphaToShared[threadIdx.x] = alphaToGlobal[threadIdx.x];
        indexOfShared[threadIdx.x] = indexOfGlobal[threadIdx.x];
        if((threadIdx.x) <16){
            inputBlock[threadIdx.x] = inputArray[threadIdx.x];
        }
    }
    __syncthreads();

    if( globalIndex <  (4 * arraySize) && ( globalIndex >= 4*(1<<(iteration)) )) 
    {
        int remainder = (globalIndex % 4);
        int vectorIndex = (globalIndex - 4*(1<<(iteration))) - remainder;
        uint8_t cipherSubBlock[4];
        for(int s = 0 ; s < 4 ; ++s)
        {
            cipherSubBlock[s] = alphaBlock[(vectorIndex + s)];
        }
        uint8_t result = 0;
        if( inputBlock[remainder*4 + 0]!= 0 && cipherSubBlock[0]!= 0){
            result ^= alphaToShared[(uint32_t(indexOfShared[cipherSubBlock[0]]) + uint32_t(indexOfShared[inputBlock[remainder * 4 + 0]]))%255];
        }
        if( inputBlock[remainder*4 + 1]!= 0 && cipherSubBlock[1]!= 0){
            result ^= alphaToShared[(uint32_t(indexOfShared[cipherSubBlock[1]]) + uint32_t(indexOfShared[inputBlock[remainder * 4 + 1]]))%255];
        }
        if( inputBlock[remainder*4 + 2]!= 0 && cipherSubBlock[2]!= 0){
            result ^= alphaToShared[(uint32_t(indexOfShared[cipherSubBlock[2]]) + uint32_t(indexOfShared[inputBlock[remainder * 4 + 2]]))%255];
        }
        if( inputBlock[remainder*4 + 3]!= 0 && cipherSubBlock[3]!= 0){
            result ^= alphaToShared[(uint32_t(indexOfShared[cipherSubBlock[3]]) + uint32_t(indexOfShared[inputBlock[remainder * 4 + 3]]))%255];
        }
        cipherBlock[globalIndex]  ^= result;
    }
}



__global__ void kernelFunc2(uint8_t* inputArray, uint8_t* alphaLookup, uint8_t* indexLookup)
{
    __shared__ uint8_t sharedInputArray[4][4];

    sharedInputArray[threadIdx.y][threadIdx.x] = inputArray[threadIdx.y*4 + threadIdx.x];

    __syncthreads();

    uint8_t xorResult = 0;

    #pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        uint8_t sharedValX = sharedInputArray[i][threadIdx.x];
        uint8_t sharedValY = sharedInputArray[threadIdx.y][i];
        xorResult ^= (sharedValX && sharedValY) ? alphaLookup[(uint32_t(indexLookup[sharedValX]) + uint32_t(indexLookup[sharedValY]))%255] : 0;
    }

    inputArray[threadIdx.y*4 + threadIdx.x] = xorResult;
}

__global__ void kernelFunc3(  uint8_t* inputArray , uint8_t* outputArray, int iteration , int arraySize)
{
	int globalIndex = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
	if( globalIndex < (4 * arraySize) && globalIndex >= 4 * (1 << iteration))
	{
		outputArray[globalIndex] = inputArray[globalIndex];
	}
}


void gpuKernel(  const uint8_t* const a, const uint8_t* const matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
    uint8_t* ab;
 	uint8_t* cb;
	uint8_t* matrixb;
	uint8_t* alpha_to_g;
	uint8_t* index_of_g;
    HANDLE_ERROR(cudaMalloc((void**)&ab , 4 * n * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&cb , 4 * n * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&matrixb , 4 * 4 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_to_g , 256 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&index_of_g , 256 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMemcpy(ab , a , 4 * n * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cb , a , 4 * n * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_to_g , alpha_to , 256 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(index_of_g , index_of , 256 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(matrixb  , matrix , 4 * 4 * (sizeof(uint8_t)) , cudaMemcpyHostToDevice));
	dim3 blockSize(1024);
    dim3 gridSize(512 , 512);
	dim3 block ( 4 , 4  );
	for(int i = 0 ; i < m ; ++i)
	{
		kernelFunc<<< gridSize , blockSize >>>(cb , ab , matrixb , i , n , alpha_to_g , index_of_g ) ;
		kernelFunc3<<< gridSize , blockSize >>>(cb , ab , i , n );
		kernelFunc2<<< 1 , block >>>( matrixb , alpha_to_g , index_of_g );
	}
	HANDLE_ERROR(cudaMemcpy(c , cb , 4 * n * sizeof(uint8_t) , cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(ab));
	HANDLE_ERROR(cudaFree(cb));
	HANDLE_ERROR(cudaFree(matrixb));
	HANDLE_ERROR(cudaFree(index_of_g));
	HANDLE_ERROR(cudaFree(alpha_to_g));
 }
