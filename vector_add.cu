#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define N 1000000
#define THREAD_PER_BLOCK 1000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b)
{
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    printf("ThreadID : %d, BlockDim : %d, BlockIdx : %d\n", threadIdx.x, blockDim.x, blockIdx.x);

    if (index < N)
        out[index] = a[index] + b[index];
}

int main()
{
    cudaProfilerStart();

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize host arrays
    for (long i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, sizeof(float) * N);
    cudaMalloc((void **)&d_b, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel
    vector_add<<<N / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(d_out, d_a, d_b);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (long i = 0; i < N; i++)
    {
        // assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
        assert(out[i] == 3.0);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    cudaProfilerStop();
    cudaDeviceReset();

    return 0;
}
