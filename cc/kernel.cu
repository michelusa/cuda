
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>



using namespace std;


__global__ void GPU_mt_info()
{
    printf("Block idx: %d | thread idx: %d\n", blockIdx.x, threadIdx.x);
}

void mt_info(int block_sz, int th_sz)
{
    GPU_mt_info << <block_sz, th_sz >> > ();
}


