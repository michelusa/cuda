#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using namespace std;
__global__ void GPU_mt_info()
{
	printf("Block idx: %d | thread idx: %d\n", blockIdx.x, threadIdx.x);
}



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

   GPU_increment_number <<< 1, 1 >>> (buffer, 5);
   
    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, buffer, BUF_SIZE, cudaMemcpyDeviceToHost);
    printf("Incrementing result is %d \n", host_buffer[0]);

    cudaFree(buffer);
    free(host_buffer);
}


int main()
{

	int device_count;

	cudaGetDeviceCount(&device_count);
	cout << "CUDA device count " << device_count << endl;

	cudaDeviceProp device_props;
	cudaGetDeviceProperties(&device_props, 0);
	cout << "CUDA first device name: " << device_props.name << endl;

	int driver_version;
	cudaDriverGetVersion(&driver_version);
	cout << "CUDA driver version: " << driver_version << endl;

	int runtime_version;
	cudaRuntimeGetVersion(&runtime_version);
	cout << "CUDA runtime version: " << runtime_version << endl;

	cout << "maxThreadsPerBlock: " << device_props.maxThreadsPerBlock << endl;

	cout << "CUDA Total global mem: " << device_props.totalGlobalMem / (1048576.0) << " MB" << endl;


	cudaEvent_t start;
	cudaEvent_t end;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	GPU_mt_info << <5, 10 >> > ();
	cudaDeviceSynchronize();
	cudaEventRecord(end);
	cudaEventSynchronize(end);


	//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
	cudaEventElapsedTime(&duration, start, end);
	printf("Duration = %f ms.\n", duration);
	cudaEventDestroy(start);
	cudaEventDestroy(end);


	cudaDeviceSynchronize();


}
