
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
using namespace std;


__global__ void GPU_increment_number(int* buffer, int initial)
{
    buffer[0] = 1 + initial;
}

void simple_exchange()
{
    int* buffer;
    const auto BUF_SIZE{ 1 * sizeof(int) };

    int* host_buffer = (int*)malloc(BUF_SIZE);
    *host_buffer = 99;

    cudaMalloc((void**)&buffer, BUF_SIZE);
    cudaMemcpy(buffer, host_buffer, BUF_SIZE, cudaMemcpyHostToDevice);

    GPU_increment_number << < 1, 1 >> > (buffer, 5);

    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, buffer, BUF_SIZE, cudaMemcpyDeviceToHost);
    printf("Incrementing result is %d \n", host_buffer[0]);

    cudaFree(buffer);
    free(host_buffer);
}


