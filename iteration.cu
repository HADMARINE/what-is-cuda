#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void loop_gpu()
{
    printf("GPU Loop, NUM : %d\n", threadIdx.x);
}

void loop_cpu(int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("CPU Loop, NUM : %d\n", i);
    }
}

int main()
{
    cudaProfilerStart();

    int loop_count = 10;
    loop_cpu(loop_count);
    loop_gpu<<<1, loop_count>>>();
    cudaDeviceSynchronize();

    cudaProfilerStop();

    return 0;
}