#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define ITER 100000
#define THREAD_PER_BLOCK 10
#define PI 3.1415926535
#define RAD(X) X *(PI / 180.0)

void calculator(float *sin_arr, float *cos_arr, float *tan_arr)
{

    for (int idx = 0; idx < ITER; idx++)
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

    sin_arr = (float *)malloc(sizeof(float) * ITER);
    cos_arr = (float *)malloc(sizeof(float) * ITER);
    tan_arr = (float *)malloc(sizeof(float) * ITER);

    calculator(sin_arr, cos_arr, tan_arr);

    for (int i = 0; i < ITER; i++)
    {
        printf("sin (%d) = %f cos (%d) = %f tan (%d) = %f\n", i, sin_arr[i], i, cos_arr[i], i, tan_arr[i]);
    }

    free(sin_arr);
    free(cos_arr);
    free(tan_arr);

    cudaProfilerStop();

    return 0;
}