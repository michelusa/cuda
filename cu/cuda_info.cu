#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using namespace std;

int main(void)
{

 '
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

    cout << "maxThreadsPerBlock: " << device_props.maxThreadsPerBlock << endl;    int device;

}
