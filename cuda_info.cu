#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

int main(void)
{
    int device;
    int device_count;
    int numSMs;
    cudaGetDeviceCount(&device_count);
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    cout << "device count " << device_count << endl;
    cout << "device " << device << endl;
    cout << "numSMs " << numSMs << endl;

}
