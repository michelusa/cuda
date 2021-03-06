#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


void simple_add();
void simple_exchange();
void mt_info(int block_sz, int th_sz);

using namespace std;
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
   
    mt_info(5, 2);
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);


    //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
    cudaEventElapsedTime(&duration, start, end);
    printf("Duration = %f ms.\n", duration);
    cudaEventDestroy(start);
    cudaEventDestroy(end);


    cudaDeviceSynchronize();


    simple_exchange();
    simple_add();


}
