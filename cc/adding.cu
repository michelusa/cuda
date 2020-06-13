
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
using namespace std;


__global__ void GPU_vector_add(int* left, int* right, int* result)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = left[idx] + right[idx];
}

void simple_add()
{
    const int DIM{ 1000 };
    int* left;  cudaMalloc(&left, DIM * sizeof(int));
    int* right;  cudaMalloc(&right, DIM * sizeof(int));
    int* result;  cudaMalloc(&result, DIM * sizeof(int));

    int* main_result = new int[DIM];
    vector<int> mainv_left;  for (int i = 0; i < DIM; ++i) mainv_left.push_back(i);;
    vector<int> mainv_right;  for (int i = 0; i < DIM; ++i) mainv_right.push_back(i * 2);;

    cudaMemcpy(left, mainv_left.data(), DIM * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(right, mainv_right.data(), DIM * sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(left);     cudaFree(right);
    mainv_left.clear(); mainv_right.clear();

    const int thrds = 2;
    GPU_vector_add << <DIM / thrds, thrds >> > (left, right, result);


    cudaMemcpy(main_result, result, DIM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cout << "simple add result:\n\t";
    for (int i = 0; i < DIM; ++i) cout << *(main_result + i) << " ";
    cudaFree(result);
    delete[] main_result;
}

