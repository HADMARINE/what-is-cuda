#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void some_kernel(int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        printf("this is valid.")
    }
}

int main()
{
}