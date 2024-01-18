#include "convolution.h"

#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!


//-----------------------------------------------------------------------------

/*__global__ void kernelFunc(const float *f, const float *g, float *result, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n + n - 1 && col < n + n - 1) {
        float sum = 0.0f;

        for (int i = max(0, row - n + 1); i <= min(row, n - 1); ++i) {
            for (int j = max(0, col - n + 1); j <= min(col, n - 1); ++j) {
                int fRow = row - i;
                int fCol = col - j;
                sum += f[fRow * n + fCol] * g[i * n + j];
            }
        }

        result[row * (n + n - 1) + col] = sum;
    }
}*/


__global__ void kernelFunc(const float *f, const float *g, float *result, int n)
{
    // Define shared memory
    __shared__ float shared_f[32];
    __shared__ float shared_g[32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n + n - 1 && col < n + n - 1) {
        float sum = 0.0f;

        // Load indices of f and g to shared memory
        int fRowStart = max(0, row - n + 1);
        int fRowEnd = min(row, n - 1);
        int fColStart = max(0, col - n + 1);
        int fColEnd = min(col, n - 1);

        for (int i = fRowStart; i <= fRowEnd; ++i) {
            for (int j = fColStart; j <= fColEnd; ++j) {
                int fRow = row - i;
                int fCol = col - j;

                // Copy indices of f and g to shared memory
                shared_f[threadIdx.x] = f[fRow * n + fCol];
                shared_g[threadIdx.x] = g[i * n + j];

                sum += shared_f[threadIdx.x] * shared_g[threadIdx.x];
            }
        }

        result[row * (n + n - 1) + col] = sum;
    }
}
