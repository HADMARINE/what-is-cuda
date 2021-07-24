#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define ITER 100000
#define THREAD_PER_BLOCK 10
#define PI 3.1415926535
#define RAD(X) X *(PI / 180.0)

__global__ void calculator_kernel(float *sin_arr, float *cos_arr, float *tan_arr, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        float rad = RAD(idx);
        sin_arr[idx] = sinf(rad);
        cos_arr[idx] = cosf(rad);
        tan_arr[idx] = tanf(rad);
    }
}

int main()
{
    cudaProfilerStart();

    float *sin_arr, *cos_arr, *tan_arr;

    cudaMallocManaged((void **)&sin_arr, sizeof(float) * ITER);
    cudaMallocManaged((void **)&cos_arr, sizeof(float) * ITER);
    cudaMallocManaged((void **)&tan_arr, sizeof(float) * ITER);

    calculator_kernel<<<ITER / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(sin_arr, cos_arr, tan_arr, ITER);
    cudaDeviceSynchronize();

    for (int i = 0; i < ITER; i++)
    {
        printf("sin (%d) = %f cos (%d) = %f tan (%d) = %f\n", i, sin_arr[i], i, cos_arr[i], i, tan_arr[i]);
    }

    cudaFree(sin_arr);
    cudaFree(cos_arr);
    cudaFree(tan_arr);

    cudaProfilerStop();

    return 0;
}