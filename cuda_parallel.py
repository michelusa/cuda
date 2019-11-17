import numpy as np

from numba import vectorize

import math

@vectorize(['float32(float32, float32)'], target='cuda')
def Add_cuda(a, b):

    res = 0;
    for x in range(10):
      res = res + x*math.cos(a) + math.exp(0.3 * math.sin(b))

    return res

@vectorize(['float32(float32, float32)'], target='cpu')
def Add_reg(a, b):

    res = 0;
    for x in range(10):
        res = res + x*math.cos(a) + math.exp(0.3 * math.sin(b))

    return res

# Initialize arrays
N = 199999009
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)

C_gpu = np.empty_like(A, dtype=A.dtype)
C_reg = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU

from timeit import default_timer as timer

start_gpu = timer()
elapsed_gpu = timer() - start_gpu
C_gpu = Add_cuda(A, B);
print( "Time to add with GPU %f " % elapsed_gpu)

# Add arrays Regular

start_reg = timer()
C_reg = Add_reg(A, B);
elapsed_reg = timer() - start_reg
print( "Time to add with reg %f " % elapsed_reg)



